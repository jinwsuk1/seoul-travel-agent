import os
import json
import urllib.parse
import requests
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models  # 💡 DB 필터 검색을 위해 추가됨
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
def get_google_place_photo(restaurant_name, region=""):
    try:
        query = f"{restaurant_name} {region} 서울"

        # 장소 검색
        search_url = (
            "https://maps.googleapis.com/maps/api/place/textsearch/json"
        )

        params = {
            "query": query,
            "key": GOOGLE_API_KEY
        }

        response = requests.get(search_url, params=params)
        data = response.json()

        if not data.get("results"):
            return None

        place = data["results"][0]

        # 사진이 없는 경우
        if "photos" not in place:
            return None

        photo_ref = place["photos"][0]["photo_reference"]

        # 사진 URL 생성
        photo_url = (
            "https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth=800"
            f"&photo_reference={photo_ref}"
            f"&key={GOOGLE_API_KEY}"
        )

        return photo_url

    except Exception as e:
        print("Google Places 오류:", e)
        return None

# =====================================================================
# 1. 초기 세팅 (로컬 모델, DB, 환경 변수)
# =====================================================================
print("⏳ 시스템을 초기화하는 중입니다. 잠시만 기다려주세요...")

# .env 파일에서 환경변수를 불러옵니다.
load_dotenv() 

# 벡터 검색 엔진 및 로컬 오픈소스 LLM 세팅
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(path="./qdrant_local_db")
local_llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# =====================================================================
# 🧠 2. 지식 그래프(Knowledge Graph) 메모리 구축
# =====================================================================
print("🧠 3만 6천 건의 데이터를 기반으로 지식 그래프(KG) 온톨로지를 빌드하는 중...")
kg_df = pd.read_csv("refined_seoul_spots.csv", low_memory=False)

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

        # 안전한 지역구 필터링
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
    print(f"   🕸️ [지식 그래프 도구] '{restaurant_name}'의 온톨로지 관계망 추적 중...")
    try:
        if restaurant_name not in G:
            return "연관 대안 매장 정보 없음"
            
        district = None
        category = None
        
        for neighbor in G.successors(restaurant_name):
            edge_data = G[restaurant_name][neighbor]
            for key in edge_data:
                rel = edge_data[key].get('relation')
                if rel == 'LOCATED_IN': district = neighbor
                if rel == 'IS_A': category = neighbor
                
        if not district or not category: 
            return "연관 데이터 부족"

        alternatives = []
        for r in G.predecessors(district):
            if r != restaurant_name and G.has_edge(r, category):
                alternatives.append(r)
                if len(alternatives) >= 2: 
                    break 
                
        if alternatives:
            return f"같은 [{district}] 내 동일한 [{category}] 업종 대안 매장: " + ", ".join(alternatives)
        return f"주변 상권 내 동일 업종 대안 매장 없음"
    except Exception as e:
        return "관계망 분석 불가"

# =====================================================================
# 🖼️ [도구 2: 로컬 멀티모달 & 카카오맵 도구 (Qdrant DB 다이렉트 검색)]
# =====================================================================
def local_multimodal_tool(restaurant_name, region):

    print(
        f"🖼️ [Google Places] {restaurant_name} 이미지 검색 중..."
    )

    image_url = get_google_place_photo(
        restaurant_name,
        region
    )

    if not image_url:
        encoded_name = urllib.parse.quote(restaurant_name)

        image_url = (
            f"https://dummyimage.com/400x300/"
            f"cccccc/000000.png&text={encoded_name}"
        )

    search_query = f"{region} {restaurant_name}"

    place_url = (
        "https://www.google.com/maps/search/"
        + urllib.parse.quote(search_query)
    )

    return image_url, place_url

# =====================================================================
# ✍️ [에이전트 3: 답변 생성 에이전트 (Generator)] - GraphRAG 융합
# =====================================================================
def generator_agent(query, search_results, region):
    print("✍️ [답변 생성 에이전트] 벡터, 지식 그래프, 미디어 데이터를 융합 중...")
    
    if not search_results:
        return "조건에 맞는 맛집 정보를 DB에서 찾을 수 없습니다."

    enriched_information = []
    for idx, hit in enumerate(search_results, 1):
        name = hit.payload.get('name', '이름 모를 식당')
        context = hit.payload.get('context', '')
        
        # Qdrant 이미지 컬렉션을 조회하는 툴 실행
        image_url, place_url = local_multimodal_tool(name, region) 
        kg_inference = kg_search_tool(name)
        
        enriched_information.append(
            f"[{idx}번 식당]\n"
            f"- 상호명: {name}\n"
            f"- 핵심 정보: {context}\n"
            f"- 사진 URL: {image_url}\n"
            f"- 구글맵 URL: {place_url}\n"
            f"- 대안 매장 정보: {kg_inference}\n"
        )

    final_context = "\n".join(enriched_information)

    prompt = f"""
    당신은 제공된 [참고 정보]만을 바탕으로 답변을 작성하는 시스템입니다.
    검색된 3개의 식당을 절대로 하나로 합치지 말고, 각각 독립적으로 소개하세요.

    [작성 규칙 - 매우 중요]
    1. 검색된 모든 식당을 하나씩 아래 [출력 템플릿]에 맞춰 작성하세요.
    2. '대안 매장 정보'가 있다면 템플릿의 대안 매장 부분에 적고, 없다면 해당 줄을 생략하세요.
    3. 템플릿 외의 불필요한 인사말, 중복된 요약, 맺음말은 절대 쓰지 마세요.

    [출력 템플릿]
    ### 🍽️ [상호명]
    - **위치 및 특징**: [핵심 정보 요약]
    - **💡 GraphRAG 추천**: 만약 자리가 없다면, 지식 그래프가 추천하는 [대안 매장 정보의 식당 이름]을(를) 방문해 보세요!

    ![식당 사진]([참고 정보에 제공된 사진 경로를 1글자도 바꾸지 말고 그대로 복사해서 넣을 것. 절대 https:// 등을 임의로 붙이지 마세요.])
    * 🗺️ [구글맵에서 상세 정보 확인하기]([참고 정보에 제공된 카카오맵 URL])
    
    ---

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
    print("\n🚀 [GraphRAG & 로컬 멀티모달] 초고속 에이전트 구동!")
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