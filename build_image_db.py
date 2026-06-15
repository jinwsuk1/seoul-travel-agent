import os
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

print("🚀 멀티모달 이미지 임베딩 및 Qdrant 적재를 시작합니다...")

model = SentenceTransformer('clip-ViT-B-32')

qdrant = QdrantClient(path="./qdrant_local_db")
COLLECTION_NAME = "seoul_images"

try:
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
except Exception:
    pass

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

# 3. 매핑 데이터 로드
try:
    df = pd.read_csv("local_multimodal_mapping.csv")
except FileNotFoundError:
    print("⚠️ local_multimodal_mapping.csv 파일이 없습니다. 이전 단계를 확인하세요.")
    exit()


points = []
success_count = 0

for idx, row in df.iterrows():
    name = row['사업장명']
    img_path = row['이미지경로']
    
    if not os.path.exists(img_path):
        continue
        
    try:
        image = Image.open(img_path)
        img_vector = model.encode(image).tolist()
        
        point = PointStruct(
            id=idx,
            vector=img_vector,
            payload={
                "name": name,
                "region": row['지역구'],
                "category": row['업태구분명'],
                "image_path": img_path 
            }
        )
        points.append(point)
        success_count += 1
        
        if success_count % 100 == 0:
            print(f"   ... {success_count}개 변환 완료")
            
    except Exception as e:
        print(f"   ⚠️ {name} 이미지 처리 중 에러: {e}")

# 5. DB에 일괄 저장
if points:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"\n🎉 성공! 총 {success_count}개의 이미지 벡터가 '{COLLECTION_NAME}' 컬렉션에 저장되었습니다.")
else:
    print("\n⚠️ 저장할 이미지 데이터가 없습니다.")