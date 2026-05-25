import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. 초기 세팅 (CPU 모드로 안전하게)
client = QdrantClient(path="./qdrant_local_db")
model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "seoul_spots"
BATCH_SIZE = 500  # 500개씩 묶어서 처리!

def ingest_data(file_path):
    print("데이터를 읽어오는 중...")
    try:
        df = pd.read_csv(file_path, encoding='cp949', encoding_errors='ignore')
    except TypeError:
        df = pd.read_csv(file_path, encoding='cp949')

    # =====================================================================
    # 🧹 데이터 정제 작업 (결측치 및 폐업 제거)
    # =====================================================================
    df = df[df['영업상태명'] == '영업/정상']
    df = df[df['사업장명'].str.contains('[가-힣]', na=False)]
    df = df.drop_duplicates(subset=['사업장명', '도로명주소'])

    total_data = len(df)
    print(f"총 {total_data}개의 정제된 데이터를 발견했습니다!")

    # 🌟 [추가된 부분] 교수님 발표용 전/후 비교를 위한 CSV 추출 로직
    print("💾 정제된 데이터를 'refined_seoul_spots.csv' 파일로 저장합니다...")
    # 한글 깨짐을 방지하기 위해 utf-8-sig 인코딩 사용
    df.to_csv("refined_seoul_spots.csv", index=False, encoding="utf-8-sig")
    print("✅ CSV 파일 저장 완료! (프로젝트 폴더를 확인해보세요)")
    print("-" * 50)

    # =====================================================================
    # 🗄️ Qdrant 벡터 DB 구축 작업
    # =====================================================================
    # DB 컬렉션 초기화
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    points = []
    processed_count = 0

    print("데이터를 500개씩 묶어서 DB에 저장하기 시작합니다...")
    
    for row in df.itertuples(index=False):
        row_dict = row._asdict() 
        
        context = f"{row_dict['사업장명']}은(는) {row_dict['도로명주소']}에 위치한 {row_dict['업태구분명']}입니다."
        vector = model.encode(context).tolist()
        
        points.append(PointStruct(
            id=processed_count,
            vector=vector,
            payload={
                "name": row_dict['사업장명'],
                "address": row_dict['도로명주소'],
                "category": row_dict['업태구분명'],
                "context": context
            }
        ))
        
        processed_count += 1
        
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   ... {processed_count}/{total_data} 개 저장 완료")
            points = []
            
    # 남은 데이터 처리
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"   ... {processed_count}/{total_data} 개 저장 완료")
        
    print("🎉 벡터 DB 구축이 완벽하게 끝났습니다!")

if __name__ == "__main__":
    # 원본 CSV 파일의 이름을 여기에 맞춰주세요 (예: seoul_spots_raw.csv)
    # 다운로드하셨던 원본 데이터 파일명을 입력하면 됩니다.
    ingest_data("서울시 휴게음식점 인허가 정보.csv")