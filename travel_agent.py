import os
import json
import urllib.parse
import requests
from dotenv import load_dotenv  # 🌟 환경 변수 로드를 위해 추가
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. 초기 세팅 (로컬 모델, DB 및 환경 변수 연동)
# =====================================================================
print("⏳ 모델과 DB를 불러오는 중입니다. 잠시만 기다려주세요...")

# 🌟 .env 파일에 저장된 환경 변수를 시스템에 등록합니다.
load_dotenv() 

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(path="./qdrant_local_db")

# 로컬 오픈소스 모델(Qwen 3B) 사용
local_llm = ChatOllama(model="qwen2.5:3b", temperature=0)

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
# 🖼️ [외부 도구: 카카오 API 멀티모달 & 메타데이터 검색기]
# =====================================================================
def kakao_search_tool(restaurant_name, region):
    """
    카카오 '이미지 검색 API'와 '로컬 검색 API'를 사용하여 실제 사진과 상세 링크를 가져옵니다.
    """
    print(f"   📡 [멀티모달 도구] 카카오 API로 '{restaurant_name}'의 실제 사진과 정보 수집 중...")
    
    # 🌟 [수정 완료] 하드코딩을 지우고 .env 파일에서 키를 안전하게 읽어옵니다.
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
    
    if not KAKAO_API_KEY:
        print("   ⚠️ 경고: .env 파일에서 KAKAO_API_KEY를 찾을 수 없습니다.")
        encoded_name = urllib.parse.quote(restaurant_name)
        return f"https://dummyimage.com/400x300/cccccc/000000.png&text={encoded_name}", "API 키 없음"
        
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    search_query = f"{region} {restaurant_name}".strip()
    
    # 1. 이미지 검색 API
    encoded_name = urllib.parse.quote(restaurant_name)
    image_url = f"https://dummyimage.com/400x300/cccccc/000000.png&text={encoded_name}"
    
    try:
        img_url = "https://dapi.kakao.com/v2/search/image"
        img_res = requests.get(img_url, headers=headers, params={"query": search_query, "size": 1})
        if img_res.status_code == 200 and img_res.json()['documents']:
            image_url = img_res.json()['documents'][0]['image_url']
    except Exception as e:
        print(f"   ⚠️ 이미지 검색 실패: {e}")

    # 2. 로컬(장소) 검색 API (카카오맵 링크)
    place_url = "링크를 찾을 수 없습니다."
    try:
        loc_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        loc_res = requests.get(loc_url, headers=headers, params={"query": search_query, "size": 1})
        if loc_res.status_code == 200 and loc_res.json()['documents']:
            place_url = loc_res.json()['documents'][0]['place_url']
    except Exception as e:
        print(f"   ⚠️ 장소 검색 실패: {e}")

    return image_url, place_url

# =====================================================================
# ✍️ [에이전트 3: 답변 생성 에이전트 (Generator)]
# =====================================================================
def generator_agent(query, search_results, region):
    print("✍️ [답변 생성 에이전트] 텍스트와 시각 데이터를 융합하여 답변 작성 중...")
    if not search_results:
        return "조건에 맞는 맛집 정보를 DB에서 찾을 수 없습니다."

    enriched_information = []
    for hit in search_results:
        name = hit.payload.get('name', '이름 모를 식당')
        context = hit.payload.get('context', '')
        
        image_url, place_url = kakao_search_tool(name, region) 
        
        enriched_information.append(
            f"식당 정보: {context}\n"
            f"사진 링크: {image_url}\n"
            f"상세 지도 링크: {place_url}"
        )

    final_context = "\n\n".join(enriched_information)

    prompt = f"""
    너는 서울 맛집 전문가야. 아래 제공된 [참고 정보]에 있는 식당만 이용해서 질문에 친절하게 답해줘.
    
    [필수 양식]
    반드시 각 식당의 소개 끝에 아래 마크다운 양식을 똑같이 적용해서 사진과 지도 링크를 넣어줘.
    ![사진](사진링크)
    * [카카오맵에서 상세 정보 및 리뷰 보기](상세 지도 링크)

    없는 식당을 지어내거나 다른 지역을 말하면 절대 안 돼.

    [참고 정보]
    {final_context}

    [질문]
    {query}
    """
    response = local_llm.invoke([HumanMessage(content=prompt)])
    return response.content

# =====================================================================
# ⚙️ [메인 오케스트레이션 (에이전트 지휘)]
# =====================================================================
def run_agentic_rag(query):
    region, keyword = router_agent(query)
    results = retriever_agent(region, keyword)
    final_answer = generator_agent(query, results, region)
    return final_answer

if __name__ == "__main__":
    print("\n🚀 보안 설정이 완료된 멀티모달 에이전트 맛집 시스템 시작!")
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
        print("[최종 AI 답변 (멀티모달)]:\n\n", answer)
        print("=" * 50 + "\n")