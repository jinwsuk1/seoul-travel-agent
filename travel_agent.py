import os
from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. 최신 Gemini 클라이언트 세팅 (기존 .env에 적어둔 키 값을 그대로 불러옵니다)
api_key = os.getenv("OPENAI_API_KEY") # .env에 저장된 변수명에 맞게 설정
if not api_key:
    print("⚠️ 에러: .env 파일에 API 키가 설정되지 않았습니다.")

client_gemini = genai.Client(api_key=api_key)

# 2. 벡터 DB 및 임베딩 모델 세팅
qdrant = QdrantClient(path="./qdrant_local_db")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_agent(query):
    # 1. 질문을 숫자로 변환 (벡터화)
    query_vector = embed_model.encode(query).tolist()

    # 2. 벡터 DB 검색
    try:
        response = qdrant.query_points(collection_name="seoul_spots", query=query_vector, limit=3)
        search_results = response.points
    except Exception:
        search_results = qdrant.search(collection_name="seoul_spots", query_vector=query_vector, limit=3)

    # 3. 참고 정보 조립
    if not search_results:
        return "죄송합니다. DB에서 관련 맛집 정보를 찾을 수 없습니다."
        
    information = "\n".join([hit.payload.get('context', '') for hit in search_results])

    # 4. 프롬프트 작성
    prompt = f"""
    너는 서울 맛집 전문가야. 아래 제공된 정보를 바탕으로 사용자의 질문에 친절하게 답변해줘.
    
    [참고 정보]
    {information}
    
    [질문]
    {query}
    """
    
    # 5. 최신 문법으로 Gemini에게 답변 요청
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print("🚀 서울 맛집 Gemini 에이전트 준비 완료!")
    print("💡 (종료하려면 '종료', 'exit', 'q' 중 하나를 입력하세요.)\n")
    
    # 무한 반복문 시작
    while True:
        user_input = input("어떤 곳을 찾으시나요? : ")
        
        # 사용자가 종료를 원할 때 루프 탈출
        if user_input.lower() in ['종료', 'exit', 'q', 'quit']:
            print("👋 AI 에이전트를 종료합니다. 수고하셨습니다!")
            break
            
        # 아무것도 입력하지 않고 엔터를 쳤을 때 방지
        if not user_input.strip():
            continue
            
        print("\n[AI 답변]:")
        print(ask_agent(user_input))
        print("\n" + "="*50 + "\n") # 다음 질문과 구분하기 위한 선