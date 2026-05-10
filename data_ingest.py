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

    # 데이터 정제 (head(500) 제거 -> 전체 데이터 대상)
    df = df[df['영업상태명'] == '영업/정상']
    df = df[df['사업장명'].str.contains('[가-힣]', na=False)]
    df = df.drop_duplicates(subset=['사업장명', '도로명주소'])

    total_data = len(df)
    print(f"총 {total_data}개의 정제된 데이터를 발견했습니다!")

    # DB 컬렉션 초기화
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    points = []
    processed_count = 0

    print("데이터를 500개씩 묶어서 DB에 저장하기 시작합니다...")
    
    # iterrows() 대신 itertuples()를 쓰면 파이썬 메모리 효율이 조금 더 좋습니다.
    for row in df.itertuples(index=False):
        # 행 데이터를 딕셔너리처럼 쓰기 위해 변환 (pandas 버전에 따라 안전하게)
        row_dict = row._asdict() 
        
        context = f"{row_dict['사업장명']}은(는) {row_dict['도로명주소']}에 위치한 {row_dict['업태구분명']}입니다."
        vector = model.encode(context).tolist()
        
        points.append(PointStruct(
            id=processed_count, # 고유 ID 부여
            vector=vector,
            payload={
                "name": row_dict['사업장명'],
                "address": row_dict['도로명주소'],
                "category": row_dict['업태구분명'],
                "context": context
            }
        ))
        
        processed_count += 1
        
        # 💡 핵심: points 리스트에 데이터가 500개 쌓일 때마다 DB에 쏘고 리스트를 싹 비웁니다!
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"진행 상황: {processed_count} / {total_data} 개 저장 완료...")
            points = [] # 메모리 비우기 (초기화)
            
    # 마지막으로 500개가 안 채워지고 남은 짜투리 데이터들 처리
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"진행 상황: {processed_count} / {total_data} 개 저장 완료...")

    print("✅ 전체 데이터가 성공적으로 벡터 DB에 저장되었습니다!")

if __name__ == "__main__":
    ingest_data("서울시 휴게음식점 인허가 정보.csv")