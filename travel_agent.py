import os
from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ 에러: .env 파일에 API 키가 설정되지 않았습니다.")

client_gemini = genai.Client(api_key=api_key)

qdrant = QdrantClient(path="./qdrant_local_db")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_agent(query):
    query_vector = embed_model.encode(query).tolist()

    try:
        response = qdrant.query_points(collection_name="seoul_spots", query=query_vector, limit=3)
        search_results = response.points
    except Exception:
        search_results = qdrant.search(collection_name="seoul_spots", query_vector=query_vector, limit=3)

    if not search_results:
        return "죄송합니다. DB에서 관련 맛집 정보를 찾을 수 없습니다."
        
    information = "\n".join([hit.payload.get('context', '') for hit in search_results])

    prompt = f"""
    너는 서울 맛집 전문가야. 아래 제공된 정보를 바탕으로 사용자의 질문에 친절하게 답변해줘.
    
    [참고 정보]
    {information}
    
    [질문]
    {query}
    """
    
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print("🚀 서울 맛집 Gemini 에이전트 준비 완료!")
    print("💡 (종료하려면 '종료', 'exit', 'q' 중 하나를 입력하세요.)\n")
    
    while True:
        user_input = input("어떤 곳을 찾으시나요? : ")
        
        if user_input.lower() in ['종료', 'exit', 'q', 'quit']:
            print("👋 AI 에이전트를 종료합니다. 수고하셨습니다!")
            break
            
        if not user_input.strip():
            continue
            
        print("\n[AI 답변]:")
        print(ask_agent(user_input))
        print("\n" + "="*50 + "\n")