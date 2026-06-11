import os
import sys
import re
import json
import urllib.parse
import requests
import pandas as pd
import networkx as nx
import concurrent.futures
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# 윈도우 환경 등에서 한글 및 이모지 출력 시 발생할 수 있는 인코딩 에러 방지
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stdin.encoding != 'utf-8':
        sys.stdin.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from qdrant_client.http import models  # 💡 DB 필터 검색을 위해 추가됨
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. 초기 세팅 (로컬 모델, DB, 환경 변수)
# =====================================================================
print("⏳ 시스템을 초기화하는 중입니다. 잠시만 기다려주세요...")

# .env 파일에서 환경변수를 불러옵니다.
load_dotenv() 

# 벡터 검색 엔진 및 로컬 오픈소스 LLM 세팅
# ---------------------------------------------------------------------
# 🤖 로컬 벡터 임베딩 모델 및 로컬 오픈소스 LLM 인스턴스 초기화
# ---------------------------------------------------------------------
# - embed_model: 자연어 질의를 고차원 벡터로 변환하기 위한 SentenceTransformer 모델
# - qdrant: 로컬 파일 기반(QdrantClient path)의 경량 벡터 데이터베이스 엔진
# - local_llm: 로컬 Ollama 환경에서 구동되는 경량/고효율 Qwen 2.5 3B 오픈소스 거대 언어 모델
# ---------------------------------------------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(path="./qdrant_local_db")
local_llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# =====================================================================
# 🧠 2. 지식 그래프(Knowledge Graph) 메모리 구축
# =====================================================================
print("🧠 3만 6천 건의 데이터를 기반으로 지식 그래프(KG) 온톨로지를 빌드하는 중...")
kg_df = pd.read_csv("refined_seoul_spots.csv", encoding="utf-8", low_memory=False)

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
# ---------------------------------------------------------------------
# 🕸️ NetworkX 기반의 방향성 멀티 그래프(MultiDiGraph) 인메모리 온톨로지 인덱싱
# ---------------------------------------------------------------------
# - 각 레스토랑 노드를 중심으로 주소(지역구) 및 업태(카테고리)를 관계형 엣지로 연결합니다.
# - 온톨로지 스키마 규칙:
#   1. [식당] --(LOCATED_IN)--> [지역구]  (공간적 관계 정의)
#   2. [식당] --(IS_A)--> [업태구분명]    (분류학적 관계 정의)
# ---------------------------------------------------------------------
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
# 🕵️ [에이전트 1: 의도 분석 및 라우터 에이전트 (Intent Router Agent)]
# =====================================================================
# - 역할: 사용자의 비정형 자연어 질의에서 검색 필터로 작용할 '지역구'와 '검색 키워드'를 정밀 추출합니다.
# - 원리: Local LLM에 엄격한 JSON 제약 조건 프롬프트를 주입하여 구조화된 엔티티(Entity) 정보를 리턴받습니다.
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
# 🔍 [에이전트 2: 벡터 검색 및 필터링 에이전트 (Retriever Agent)]
# =====================================================================
# - 역할: 임베딩 모델로 질의 키워드를 고차원 벡터로 인코딩한 후, 벡터 DB 유사도 검색을 수행합니다.
# - 원리: Qdrant 유사도 스코어 기준 Top 500개의 맛집 후보군을 1차 스캔한 후,
#         의도 분석 에이전트가 도출한 공간 필터(지역구 정보)를 적용해 정확히 매칭되는 상위 3개 점포를 최종 선정합니다.
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
# 🕸️ [도구 1: 지식 그래프 추론 및 연관 매장 탐색 도구 (KG Traversal Tool)]
# =====================================================================
# - 역할: 검색된 식당과 지리적(LOCATED_IN), 업태분류적(IS_A) 관계를 공유하는 대체 맛집(대안 매장)을 탐색합니다.
# - 원리: NetworkX 그래프 상에서 [현재 맛집 노드] -> [지역구/업태 노드] -> [동일한 엣지 공유 노드] 순으로
#         역방향 링크(predecessors)를 추적하여 최적의 대안 점포를 GraphRAG 추천 정보로 제안합니다.
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
# 🖼️ [도구 2: 이미지 도구 (Qdrant DB 다이렉트 검색)]
# =====================================================================
# =====================================================================
# 🖼️ [서브 도구: 로컬 폰트 렌더링 기반 Base64 플레이스홀더 생성기]
# =====================================================================
# - 역할: 외부 이미지 서버가 차단되었거나 한글 폰트를 정상 지원하지 못해 발생하는 폰트 상자 깨짐([][])을 우회합니다.
# - 원리: Pillow 라이브러리로 400x300 크기의 라이트 그레이 캔버스를 메모리에 열고,
#         Windows 시스템 맑은 고딕(malgun.ttf) 한글 폰트를 동적으로 맵핑하여 텍스트를 그린 뒤 PNG 바이트를 
#         Base64 Data URI 문자열로 즉각 인코딩하여 반환합니다.
# =====================================================================
def generate_local_placeholder(restaurant_name):
    try:
        # 400x300 크기의 에스테틱한 그레이 배경 생성
        img = Image.new('RGB', (400, 300), color='#f3f4f6')
        draw = ImageDraw.Draw(img)
        
        # Windows의 맑은 고딕 폰트 경로 사용
        font_path = "C:\\Windows\\Fonts\\malgun.ttf"
        try:
            font = ImageFont.truetype(font_path, 24)
        except Exception:
            font = ImageFont.load_default()
            
        # 텍스트 바운딩 박스를 계산하여 중앙 배치
        try:
            bbox = draw.textbbox((0, 0), restaurant_name, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(restaurant_name, font=font)
            
        x = (400 - text_w) / 2
        y = (300 - text_h) / 2
        
        # 둥근 테두리와 텍스트 렌더링
        draw.rounded_rectangle([20, 20, 380, 280], radius=10, outline="#d1d5db", width=2)
        draw.text((x, y), restaurant_name, fill='#374151', font=font)
        
        # 바이너리 바이트 변환 후 Base64 인코딩
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        byte_im = buf.getvalue()
        return "data:image/png;base64," + base64.b64encode(byte_im).decode('utf-8')
    except Exception as e:
        print(f"      ⚠️ 로컬 플레이스홀더 생성 실패: {e}")
        encoded_name = urllib.parse.quote(restaurant_name)
        return f"https://placehold.co/400x300/cccccc/000000/png?text={encoded_name}"

# =====================================================================
# 📰 [도구 2-1: 네이버 실시간 이미지 검색 API 및 뉴스/방송사 필터링 도구]
# =====================================================================
# - 역할: 인터넷상에 올라와 있는 해당 맛집의 실제 전경 및 실물 음식 사진 URL을 실시간 획득합니다.
# - 필터링 정책: 단순히 1순위 이미지를 긁어올 경우, 기사 본문과 관계없는 방송 자막 및 출연자 얼굴 캡처가
#              노출되는 심각한 품질 저하가 있습니다. 이를 차단하고자 10개(`display=10`) 후보군을 수집한 뒤
#              뉴스/방송 매체 도메인을 블랙리스트로 걸러내고, 실제 현장 사진이 기록된 블로그/포스트/SNS의 이미지를 우선 반환합니다.
# =====================================================================
def naver_image_search(restaurant_name, region):
    # 특수문자 제거
    clean_name = re.sub(r'[^\w\s가-힣]', ' ', restaurant_name).strip()
    client_id = os.getenv("NAVER_CLIENT_ID", "")
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "")
    
    if not client_id or not client_secret:
        return ""
        
    # 방송 캡처 및 뉴스 자료 화면을 피하기 위한 제외 키워드/도메인 블랙리스트
    NEWS_DOMAINS = [
        "imgnews.naver.net", "news.naver.com", "chosun.com", "joongang.com", "donga.com",
        "sbs.co.kr", "kbs.co.kr", "mbc.co.kr", "news.khan.co.kr", "hani.co.kr",
        "ytn.co.kr", "jtbc.co.kr", "seoul.co.kr", "kmib.co.kr", "hankyung.com",
        "mk.co.kr", "mt.co.kr", "fnnews.com", "asiae.co.kr", "heraldcorp.com",
        "inews24.com", "nocutnews.co.kr", "newsis.com", "yonhapnewstv.co.kr",
        "yonhapnews.co.kr", "etnews.com", "press", "media"
    ]
    
    try:
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        # 좀 더 관련성 높은 실제 사진을 얻기 위해 10개 수집
        query = f"{region} {clean_name}".strip()
        params = {"query": query, "display": 10, "sort": "sim"}
        res = requests.get(
            "https://openapi.naver.com/v1/search/image.json",
            headers=headers, params=params, timeout=3
        )
        res.raise_for_status()
        items = res.json().get("items", [])
        
        # 1. 1차 필터링: 뉴스/방송사 도메인을 포함하지 않고, 개인 리뷰(블로그/포스트/카페)로 유력한 도메인 우선 탐색
        for item in items:
            link = item.get("link", "")
            if not link:
                continue
                
            # 블랙리스트 도메인 체크
            is_news = any(domain in link for domain in NEWS_DOMAINS)
            if is_news:
                continue
                
            # 블로그 및 소셜 리뷰 미디어(tistory, naver blog 등) 우선 선택
            if any(ref in link for ref in ["blog", "post", "tistory", "daum", "egloos", "instagram", "facebook"]):
                return link
                
        # 2. 2차 필터링: 만개한 블로그는 아니지만 그래도 뉴스 도메인이 아닌 일반 이미지 탐색
        for item in items:
            link = item.get("link", "")
            if not link:
                continue
            is_news = any(domain in link for domain in NEWS_DOMAINS)
            if not is_news:
                return link
                
    except Exception as e:
        print(f"   ⚠️ 네이버 이미지 검색 API 오류: {e}")
        
    return ""

# =====================================================================
# 🖼️ [도구 2: 3단계 하이브리드 멀티모달 이미지 도구 (Hybrid Multimodal Tool)]
# =====================================================================
# - 역할: 에이전트 답변에 시각적인 식당 사진 요소를 제공하는 관문 역할을 수행합니다.
# - 3단계 획득 아키텍처:
#   1. [로컬 DB 스캔]: Qdrant 벡터 DB의 seoul_images 내에 물리 저장된 매칭 파일이 있다면 이를 즉각 사용.
#   2. [실시간 실제 이미지 수집]: DB에 없을 경우, 네이버 이미지 검색 API로 개인 리뷰(블로그 등)의 실사 사진 획득.
#   3. [로컬 Pillow 생성]: 네트워크 오류 및 검색 결과 부재 시, 한글 깨짐이 완벽히 방지된 로컬 Base64 더미 이미지 자동 생성.
# =====================================================================
def local_multimodal_tool(restaurant_name, region):
    print(f"   🖼️ [이미지 도구] '{restaurant_name}' 이미지 매칭 중...")
    
    # 1. DB 검색 시도
    try:
        records, _ = qdrant.scroll(
            collection_name="seoul_images",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="name",
                    match=models.MatchValue(value=restaurant_name)
                )]
            ),
            limit=1
        )
        if records:
            image_url = str(records[0].payload.get("image_path")).replace("\\", "/")
            if image_url:
                print("      ✅ DB 내 저장된 이미지 검색 성공")
                return image_url
    except Exception as e:
        print(f"      ⚠️ DB 이미지 검색 실패: {e}")
        
    # 2. 네이버 이미지 검색 API 시도
    real_image_url = naver_image_search(restaurant_name, region)
    if real_image_url:
        print(f"      ✅ 네이버 실시간 이미지 검색 성공: {real_image_url}")
        return real_image_url
        
    # 3. 로컬 Pillow Base64 생성
    print("      ⚠️ 대체 더미 이미지 동적 생성 적용 (Base64)")
    return generate_local_placeholder(restaurant_name)

# =====================================================================
# 🗺️ [도구 3: 카카오 로컬 API 기반 실시간 장소 검색 도구]
# =====================================================================
# - 역할: 카카오맵 로컬 데이터베이스에 실시간 쿼리하여 실제 도로명 주소, 전화번호 및 공식 상세 매장 링크를 수집합니다.
# - 개체명 명확화 (Entity Resolution): 지식 그래프 상에 정제되지 않았던 원본 상호명(예: '카페,브릭')을 정규식으로 전처리한 후,
#                                   카카오맵 상의 실제 매장명(예: '브릭커피')을 획득하여 네이버 지도 오매칭을 해결하는 기틀이 됩니다.
# =====================================================================
def kakao_search_tool(restaurant_name, region):
    # 쉼표(,), 대괄호 등 검색을 방해하는 특수문자 제거 전처리 (예: '카페,브릭' -> '카페 브릭')
    clean_name = re.sub(r'[^\w\s가-힣]', ' ', restaurant_name).strip()
    print(f"   🗺️ [카카오 API 도구] '{clean_name}'(원래 상호: '{restaurant_name}') 실시간 장소 검색 중...")
    kakao_key = os.getenv("KAKAO_API_KEY", "")
    fallback_url = f"https://map.kakao.com/link/search/{urllib.parse.quote(f'{region} {clean_name}'.strip())}"
    
    if not kakao_key:
        return {"place_name": restaurant_name, "place_url": fallback_url, "address": "", "phone": "", "category": ""}
    
    try:
        headers = {"Authorization": f"KakaoAK {kakao_key}"}
        params = {"query": f"{region} {clean_name}".strip(), "size": 1}
        res = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers=headers, params=params, timeout=5
        )
        res.raise_for_status()
        docs = res.json().get("documents", [])
        if docs:
            doc = docs[0]
            real_name = doc.get('place_name', restaurant_name)
            print(f"   ✅ 카카오 검색 성공: {real_name}")
            return {
                "place_name": real_name,
                "place_url": doc.get("place_url", fallback_url),
                "address": doc.get("road_address_name") or doc.get("address_name", ""),
                "phone": doc.get("phone", ""),
                "category": doc.get("category_name", ""),
            }
    except Exception as e:
        print(f"   ⚠️ 카카오 API 오류 (fallback 사용): {e}")
    
    return {"place_name": restaurant_name, "place_url": fallback_url, "address": "", "phone": "", "category": ""}

# =====================================================================
# 📰 [도구 4: 네이버 지역 검색 API 기반 정보 매칭 도구]
# =====================================================================
# - 역할: 네이버 검색 시스템에 매칭시켜 설명(Description) 및 주소 데이터를 한 번 더 검증하고, 지도 랜딩 페이지 링크를 생성합니다.
# - 지도 쿼리 빌딩: 사용자가 지도를 편리하게 이용할 수 있도록, 네이버 검색으로 일치된 최종 명칭을 인코딩하여 
#                 직접 검색 랜딩 페이지 URL(https://map.naver.com/v5/search/...) 형식을 통일하여 생성합니다.
# =====================================================================
def naver_review_tool(restaurant_name, region):
    # 쉼표(,), 대괄호 등 검색을 방해하는 특수문자 제거 전처리
    clean_name = re.sub(r'[^\w\s가-힣]', ' ', restaurant_name).strip()
    print(f"   📰 [네이버 API 도구] '{clean_name}'(원래 상호: '{restaurant_name}') 네이버 지역 정보 검색 중...")
    client_id = os.getenv("NAVER_CLIENT_ID", "")
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "")
    
    fallback_url = f"https://map.naver.com/v5/search/{urllib.parse.quote(f'{region} {clean_name}'.strip())}"
    
    if not client_id or not client_secret:
        return {"naver_url": fallback_url, "description": "", "address": ""}
    
    try:
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": f"{region} {clean_name}".strip(), "display": 1}
        res = requests.get(
            "https://openapi.naver.com/v1/search/local.json",
            headers=headers, params=params, timeout=5
        )
        res.raise_for_status()
        items = res.json().get("items", [])
        if items:
            item = items[0]
            title = re.sub('<[^<]+?>', '', item.get("title", ""))
            print(f"   ✅ 네이버 검색 성공: {title}")
            # 사용자가 네이버 지도 바로가기를 편리하게 사용하도록 지도 검색 결과 링크를 반환
            map_url = f"https://map.naver.com/v5/search/{urllib.parse.quote(f'{region} {title}'.strip())}"
            return {
                "naver_url": map_url,
                "description": item.get("description", ""),
                "address": item.get("roadAddress") or item.get("address", ""),
            }
    except Exception as e:
        print(f"   ⚠️ 네이버 API 오류: {e}")
    
    return {"naver_url": fallback_url, "description": "", "address": ""}

# ---------------------------------------------------------------------
# ⛓️ [도구 연쇄 오케스트레이션: API Chaining 수집 엔진]
# ---------------------------------------------------------------------
# - 역할: 개별 식당 단위로 독립적인 도구 연쇄 호출 흐름을 오케스트레이션합니다.
# - 동작 흐름:
#   1. Qdrant DB 및 네이버 이미지 수집은 Qdrant 동시성 파일 락 우회를 위해 순차적 실행.
#   2. 내부 스레드 풀을 임시 열어, 지식 그래프(KG) 추적과 카카오 API 조회를 병렬로 수행 (지연 최소화).
#   3. [API Chaining 핵심]: 카카오 API 검색 결과로 도출된 정확한 실명(place_name)을 쿼리로 변환하여
#                         네이버 지역 정보 API를 연달아 호출함으로써 상호 불일치(오매칭)를 완전히 해결.
# ---------------------------------------------------------------------
def fetch_restaurant_data_chained(idx, hit, region):
    name = hit.payload.get('name', '이름 모를 식당')
    context = hit.payload.get('context', '')
    
    # 1. Qdrant 로컬 DB 및 네이버 이미지 조회를 안전하게 순차 실행
    image_url = local_multimodal_tool(name, region)
    
    # 2. 지식 그래프 및 카카오 API 조회를 병렬 스레드 풀로 1차 수집
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_executor:
        f_kg  = inner_executor.submit(kg_search_tool, name)
        f_kak = inner_executor.submit(kakao_search_tool, name, region)
        
        kg_inference = f_kg.result()
        kakao_info   = f_kak.result()
        
    # 3. 카카오 검색을 통해 확인된 실제 상호명을 쿼리로 네이버 API 조회 (Chaining 기법)
    naver_query = kakao_info.get("place_name", name)
    naver_info = naver_review_tool(naver_query, region)
    
    return {
        "idx": idx,
        "name": name,
        "context": context,
        "image_url": image_url,
        "kg_inference": kg_inference,
        "kakao_info": kakao_info,
        "naver_info": naver_info
    }

# =====================================================================
# ✍️ [에이전트 3: 답변 생성 및 포스트프로세싱 에이전트 (Generator Agent)]
# =====================================================================
# - 역할: 수집된 지식 그래프 관계 정보, 이미지 데이터, 카카오/네이버 지리적 링크를 융합하여 마크다운 답변을 빌드합니다.
# - 고성능 2계층 병렬 오케스트레이션:
#   - 3개 맛집 각각의 Chaining 수집기(fetch_restaurant_data_chained)를 최상위 ThreadPoolExecutor에서 
#     동시에 병렬(max_workers=3) 구동하여, 외부 네트워크 요청 대기 시간을 최소화합니다.
# - LLM 생성 지연 극복 (하이브리드 조립):
#   - LLM은 맛집의 특징 요약글만 작성하게 통제하여 텍스트 추론 토큰 크기를 비약적으로 축소했습니다.
#   - 이미지 출력, 하이퍼링크, 주소, 전화번호 등의 정적 마크다운 요소는 파이썬 코드가 직렬 조립하여 결합함으로써
#     LLM의 오버헤드를 줄이고, 엑스박스 발생이나 문법 파손을 원천 차단했습니다.
# =====================================================================
def generator_agent(query, search_results, region):
    print("✍️ [답변 생성 에이전트] 벡터, 지식 그래프, 미디어 데이터를 융합 중...")
    
    if not search_results:
        return "조건에 맞는 맛집 정보를 DB에서 찾을 수 없습니다."

    # 1. 3개 맛집 각각에 대해 chained 수집기를 최상위 스레드 풀에서 동시 처리
    enriched_data_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_results)) as executor:
        tasks = [executor.submit(fetch_restaurant_data_chained, idx, hit, region) 
                 for idx, hit in enumerate(search_results, 1)]
        enriched_data_list = [t.result() for t in tasks]

    # 2. LLM 요약용 컨텍스트 구성
    llm_context_list = []
    for data in enriched_data_list:
        llm_context_list.append(
            f"[식당명: {data['name']}]\n"
            f"- 핵심 설명: {data['context']}\n"
            f"- 네이버 설명: {data['naver_info'].get('description', '')}\n"
        )
    final_context = "\n".join(llm_context_list)

    # 3. LLM에게 특징 요약글만 짧게 작성하도록 유도 (추론 지연 시간 절반 이하 단축)
    prompt = f"""
    당신은 제공된 [참고 정보]만을 바탕으로 각 식당의 '위치 및 특징' 요약글을 작성하는 시스템입니다.
    불필요한 인사말, 전체 요약, 마크다운 이미지/지도 링크는 절대 쓰지 말고, 아래 [출력 형식]에 맞게만 작성하세요.
    반드시 각 식당에 대해 개별적인 문단으로 요약해야 합니다.

    [출력 형식]
    - 식당명: [상호명]
    - 요약: [위치 및 특징에 대한 자연스러운 요약 (2-3문장)]
    ---

    [참고 정보]
    {final_context}

    [질문]
    {query}
    """
    
    response = local_llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()
    
    # 4. LLM 요약 결과 파싱
    summaries = {}
    current_restaurant = None
    
    for line in llm_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- 식당명:") or line.startswith("식당명:"):
            current_restaurant = line.split(":", 1)[1].strip().replace("[", "").replace("]", "")
        elif (line.startswith("- 요약:") or line.startswith("요약:")) and current_restaurant:
            summaries[current_restaurant] = line.split(":", 1)[1].strip()
            
    # 5. 파이썬 마크다운 최종 조립 (엑스박스 완벽 차단 및 링크 원본 보존)
    final_output_parts = []
    for data in enriched_data_list:
        name = data["name"]
        
        # LLM 요약 매칭 (없을 경우 context 기반 Fallback)
        summary = summaries.get(name)
        if not summary:
            for k, v in summaries.items():
                if k in name or name in k:
                    summary = v
                    break
        if not summary:
            summary = f"{data['context']} {data['naver_info'].get('description', '')}".strip()

        # 대안 매장 문구
        kg_text = ""
        if data["kg_inference"] and "정보 없음" not in data["kg_inference"] and "분석 불가" not in data["kg_inference"] and "부족" not in data["kg_inference"]:
            alt_match = re.search(r"대안 매장:\s*(.*)", data["kg_inference"])
            alt_names = alt_match.group(1) if alt_match else data["kg_inference"]
            kg_text = f"- **💡 GraphRAG 추천**: 만약 자리가 없다면, {alt_names}을(를) 방문해 보세요!\n"

        # 전화번호
        phone_text = ""
        if data["kakao_info"].get("phone"):
            phone_text = f"- **전화번호**: {data['kakao_info']['phone']}\n"

        # 주소 선택
        address = data["kakao_info"].get("address") or data["naver_info"].get("address") or "주소 정보 없음"

        img_url = data["image_url"]
        kakao_url = data["kakao_info"]["place_url"]
        naver_url = data["naver_info"]["naver_url"]

        # 마크다운 생성
        restaurant_md = (
            f"### 🍽️ {name}\n"
            f"- **주소**: {address}\n"
            f"{phone_text}"
            f"- **위치 및 특징**: {summary}\n"
            f"{kg_text}\n"
            f"![식당 사진]({img_url})\n\n"
            f"* 🗺️ [카카오맵 바로가기]({kakao_url})\n"
        )
        if naver_url:
            restaurant_md += f"* 🔍 [네이버 지도 바로가기]({naver_url})\n"
            
        restaurant_md += "\n---\n"
        final_output_parts.append(restaurant_md)
        
    return "\n".join(final_output_parts)


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