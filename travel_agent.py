import os
import json
import urllib.parse
import requests
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. 초기 세팅 (로컬 모델, DB, 환경 변수)
# =====================================================================
print("⏳ 시스템을 초기화하는 중입니다. 잠시만 기다려주세요...")

# .env 파일에서 KAKAO_API_KEY 등을 불러옵니다.
load_dotenv() 

# 벡터 검색 엔진 및 로컬 오픈소스 LLM 세팅
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(path="./qdrant_local_db")
local_llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# =====================================================================
# 🧠 2. 지식 그래프(Knowledge Graph) 메모리 구축 (교수님 피드백)
# =====================================================================
print("🧠 3만 6천 건의 데이터를 기반으로 지식 그래프(KG) 온톨로지를 빌드하는 중...")
kg_df = pd.read_csv("refined_seoul_spots.csv", low_memory=False)

# 주소에서 '구' 이름만 안전하게 추출하는 함수
def get_gu_name(address):
    try:
        parts = str(address).split()
        for p in parts:
            if p.endswith('구') and len(p) < 5: 
                return p
        return '기타'
    except Exception: 
        return '기타'

kg_df['지역구'] = kg_df['도로명주소'].apply(get_gu_name)

# NetworkX 방향성 멀티 그래프 생성
G = nx.MultiDiGraph()
for row in kg_df.itertuples(index=False):
    row_dict = row._asdict()
    restaurant = row_dict['사업장명']
    district = row_dict['지역구']
    category = row_dict['업태구분명']
    
    # 온톨로지 규칙: 식당은 지역구에 '위치하고(LOCATED_IN)', 특정 업종으로 '분류됨(IS_A)'
    G.add_edge(restaurant, district, relation='LOCATED_IN')
    G.add_edge(restaurant, category, relation='IS_A')

print(f"   ✅ 지식 그래프 구축 완료! (총 노드 수: {G.number_of_nodes():,}개)")

# =====================================================================
# 🕵️ [에이전트 1: 의도 분석 에이전트 (Router)]
# =====================================================================
def router_agent(query):
    print("\n🕵️ [의도 분석 에이전트] 질문에서 지역과 키워드를 추출 중...")
    prompt = f"""
    당신은 사용자의 질문에서 검색 조건을 추출하는 분석기입니다.
    질문에서 '지역(구 이름)'과 '음식 종류(키워드)'를 찾아 아래의 엄격한 JSON 형식으로만 답변하세요. 다른 설명은 절대 추가하지 마세요.
    (예시: {{"region": "강남구", "keyword": "커피"}})
    지역이 언급되지 않았다면 region을 빈 문자열("")로 두세요.

    질문: {query}
    """
    response = local_llm.invoke([HumanMessage(content=prompt)])
    
    try:
        text = response.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(text)
        region = parsed.get("region", "")
        keyword = parsed.get("keyword", query)
        print(f"   ✅ 분석 결과 -> 지역: '{region}', 키워드: '{keyword}'")
        return region, keyword
    except Exception:
        print("   ⚠️ 분석 실패. 기본 검색으로 진행합니다.")
        return "", query

# =====================================================================
# 🔍 [에이전트 2: 검색 에이전트 (Retriever)]
# =====================================================================
def retriever_agent(region, keyword):
    print("🔍 [검색 에이전트] 벡터 DB에서 관련 식당을 찾는 중...")
    
    if not keyword or keyword.strip() == "": 
        keyword = "맛집"
        
    query_vector = embed_model.encode(keyword).tolist()
    
    try:
        try:
            points = qdrant.query_points(collection_name="seoul_spots", query=query_vector, limit=500).points
        except Exception:
            points = qdrant.search(collection_name="seoul_spots", query_vector=query_vector, limit=500)

        # 안전한 지역구 필터링 (결측치 에러 방지)
        filtered_points = []
        for pt in points:
            safe_address = str(pt.payload.get("address", ""))
            if region and "구" in region:
                if region in safe_address: 
                    filtered_points.append(pt)
            else:
                filtered_points.append(pt)
                
        final_results = filtered_points[:3]
        print(f"   ✅ 필터링 완료: {len(final_results)}개의 유효한 데이터를 찾아왔습니다.")
        return final_results
        
    except Exception as e:
        print(f"   ⚠️ 검색 오류 발생: {e}")
        return []

# =====================================================================
# 🕸️ [도구 1: 지식 그래프 추론 도구 (KG Traversal Tool)]
# =====================================================================
def kg_search_tool(restaurant_name):
    """
    지식 그래프를 탐색하여 해당 식당과 '같은 구'에 있으면서 
    '같은 카테고리'인 대체 식당을 추론해 내는 도구입니다.
    """
    print(f"   🕸️ [지식 그래프 도구] '{restaurant_name}'의 온톨로지 관계망 추적 중...")
    try:
        if restaurant_name not in G:
            return "연관 대안 매장 정보 없음"
            
        district = None
        category = None
        
        # 주어 노드(식당)에서 나가는 간선을 분석해 지역과 카테고리 파악
        for neighbor in G.successors(restaurant_name):
            edge_data = G[restaurant_name][neighbor]
            for key in edge_data:
                rel = edge_data[key].get('relation')
                if rel == 'LOCATED_IN': district = neighbor
                if rel == 'IS_A': category = neighbor
                
        if not district or not category: 
            return "연관 데이터 부족"

        # 목적어 노드(구역)를 공유하는 다른 식당 중 카테고리도 일치하는 곳 찾기
        alternatives = []
        for r in G.predecessors(district):
            if r != restaurant_name and G.has_edge(r, category):
                alternatives.append(r)
                if len(alternatives) >= 2: # 대안은 최대 2개만 추천
                    break 
                
        if alternatives:
            return f"같은 [{district}] 내 동일한 [{category}] 업종 대안 매장: " + ", ".join(alternatives)
        return f"주변 상권 내 동일 업종 대안 매장 없음"
    except Exception as e:
        return "관계망 분석 불가"

# =====================================================================
# 🖼️ [도구 2: 카카오 멀티모달 & 링크 수집 도구 (Kakao API Tool)]
# =====================================================================
def kakao_search_tool(restaurant_name, region):
    print(f"   📡 [멀티모달 도구] 카카오 API로 '{restaurant_name}'의 미디어 데이터 수집 중...")
    
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
    search_query = f"{region} {restaurant_name}".strip()
    encoded_name = urllib.parse.quote(restaurant_name)
    
    image_url = f"https://dummyimage.com/400x300/cccccc/000000.png&text={encoded_name}"
    
    if not KAKAO_API_KEY:
        return image_url, f"https://map.kakao.com/link/search/{urllib.parse.quote(search_query)}"
        
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    
    # 1. 이미지 검색 API
    try:
        img_url = "https://dapi.kakao.com/v2/search/image"
        img_res = requests.get(img_url, headers=headers, params={"query": search_query, "size": 1})
        if img_res.status_code == 200 and img_res.json()['documents']:
            image_url = img_res.json()['documents'][0]['image_url']
    except Exception as e:
        print(f"   ⚠️ 이미지 검색 실패: {e}")

    # 2. 로컬 장소 검색 API (카카오맵 링크)
    place_url = ""
    try:
        loc_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        loc_res = requests.get(loc_url, headers=headers, params={"query": search_query, "size": 1})
        if loc_res.status_code == 200 and loc_res.json()['documents']:
            place_url = loc_res.json()['documents'][0]['place_url']
    except Exception as e:
        print(f"   ⚠️ 장소 검색 실패: {e}")

    # 카카오맵 검색 실패 시 직접 URL 검색결과로 이동하는 안전망 처리
    if not place_url:
        place_url = f"https://map.kakao.com/link/search/{urllib.parse.quote(search_query)}"

    return image_url, place_url

# =====================================================================
# ✍️ [에이전트 3: 답변 생성 에이전트 (Generator)] - GraphRAG 융합
# =====================================================================
def generator_agent(query, search_results, region):
    print("✍️ [답변 생성 에이전트] 벡터, 지식 그래프, 미디어 데이터를 융합 중...")
    
    if not search_results:
        return "조건에 맞는 맛집 정보를 DB에서 찾을 수 없습니다."

    enriched_information = []
    for hit in search_results:
        name = hit.payload.get('name', '이름 모를 식당')
        context = hit.payload.get('context', '')
        
        # 외부 도구 1: 이미지/링크 API 호출
        image_url, place_url = kakao_search_tool(name, region) 
        
        # 내부 도구 2: 지식 그래프 온톨로지 추론 호출 (GraphRAG 연동 핵심!)
        kg_inference = kg_search_tool(name)
        
        enriched_information.append(
            f"식당 핵심 정보: {context}\n"
            f"사진 원격 주소: {image_url}\n"
            f"카카오맵 주소: {place_url}\n"
            f"온톨로지 관계망 기반 대안 매장: {kg_inference}"
        )

    final_context = "\n\n".join(enriched_information)

    prompt = f"""
    너는 서울 맛집 전문가야. 아래 제공된 [참고 정보]에 있는 식당 정보와 온톨로지 관계망 정보를 활용해 질문에 답해줘.
    이마트, 홈플러스 등 대형마트나 PC방은 식당 추천에서 배제해줘.
    
    [답변 가이드]
    1. 각 식당의 위치와 특징을 친절하게 설명해줘.
    2. 각 식당 설명 바로 아래에 '온톨로지 관계망 기반 대안 매장'에 적힌 대체 식당도 "만약 자리가 없다면 같은 지역구 내 동일 업종인 OOO 매장도 지식 그래프 기반으로 함께 추천해 드립니다"라는 문맥으로 자연스럽게 소개해줘.
    3. 반드시 아래의 마크다운 형식을 철저히 지켜서 사진과 링크가 출력되게 해줘.
    
    ![사진](사진_원격_주소_값)
    * [카카오맵에서 상세 정보 및 리뷰 보기](카카오맵_주소_값)

    [참고 정보]
    {final_context}

    [질문]
    {query}
    """
    response = local_llm.invoke([HumanMessage(content=prompt)])
    return response.content

# =====================================================================
# ⚙️ [메인 오케스트레이션]
# =====================================================================
def run_agentic_rag(query):
    region, keyword = router_agent(query)
    results = retriever_agent(region, keyword)
    final_answer = generator_agent(query, results, region)
    return final_answer

if __name__ == "__main__":
    print("\n🚀 [GraphRAG 완성형] 지식 그래프 및 멀티모달 통합 에이전트 구동!")
    print("💡 (종료하려면 '종료', 'exit', 'q' 중 하나를 입력하세요.)\n")
    
    while True:
        user_input = input("어떤 곳을 찾으시나요? : ")
        
        if user_input.lower() in ['종료', 'exit', 'q', 'quit']:
            print("👋 시스템을 종료합니다.")
            break
            
        if not user_input.strip(): 
            continue
        
        print("\n" + "-"*50)
        answer = run_agentic_rag(user_input)
        print("-" * 50)
        print("[최종 AI 답변 (GraphRAG)]:\n\n", answer)
        print("=" * 50 + "\n")