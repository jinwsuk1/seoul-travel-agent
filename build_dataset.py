import os
import time
import requests
import pandas as pd
import re
from dotenv import load_dotenv

print("🚀 [데이터 수집기 V2] 지역/업종 밸런싱 멀티모달 데이터셋 구축 시작...")

# 1. 환경 변수 및 폴더 세팅
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
if not KAKAO_API_KEY:
    print("⚠️ KAKAO_API_KEY가 없습니다. .env 파일을 확인해주세요.")
    exit()

HEADERS = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
IMAGE_FOLDER = "./multimodal_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# 2. 원본 데이터 불러오기 및 섞기 (랜덤 셔플)
try:
    df = pd.read_csv("refined_seoul_spots.csv", low_memory=False)
except FileNotFoundError:
    print("⚠️ refined_seoul_spots.csv 파일을 찾을 수 없습니다.")
    exit()

# 전체 데이터를 무작위로 섞습니다.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


TARGET_COUNT = 1000
MAX_PER_DISTRICT = 50   
MAX_PER_CATEGORY = 150  

success_count = 0
district_counts = {}
category_counts = {}
mapping_data = []

print(f"🎯 목표치: {TARGET_COUNT}개 다운로드 완료할 때까지 검색합니다.\n")


for idx, row in df.iterrows():
    if success_count >= TARGET_COUNT:
        break 

    name = str(row['사업장명']).strip()
    category = str(row['업태구분명']).strip()
    
    match = re.search(r'([가-힣]+구)', str(row['도로명주소']))
    region = match.group(1) if match else "기타"
    
    if region == "기타":
        continue

    # 💡 밸런싱 검사: 지역구나 업종이 쿼터를 초과하면 과감히 스킵!
    if district_counts.get(region, 0) >= MAX_PER_DISTRICT:
        continue
    if category_counts.get(category, 0) >= MAX_PER_CATEGORY:
        continue

    search_query = f"{region} {name}"
    
    try:
        url = "https://dapi.kakao.com/v2/search/image"
        res = requests.get(url, headers=HEADERS, params={"query": search_query, "size": 1})
        
        if res.status_code == 200 and res.json()['documents']:
            img_url = res.json()['documents'][0]['image_url']
            
            # 실제 이미지 파일 다운로드 (시간 초과 5초 제한)
            img_response = requests.get(img_url, timeout=5)
            
            if img_response.status_code == 200:
                # 💡 덮어쓰기 방지: 파일명에 '지역구_상호명_인덱스'를 붙여 고유하게 만듦
                safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")
                file_path = f"{IMAGE_FOLDER}/{region}_{safe_name}_{idx}.jpg"
                
                with open(file_path, 'wb') as f:
                    f.write(img_response.content)
                
                # 매칭 데이터 기록
                mapping_data.append({
                    "사업장명": name,
                    "지역구": region,
                    "업태구분명": category,
                    "이미지경로": file_path
                })
                
                # 카운트 증가
                success_count += 1
                district_counts[region] = district_counts.get(region, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
                
                print(f"[{success_count}/{TARGET_COUNT}] 📸 저장완료 | {region} | {category} | {name}")
            else:
                pass # 이미지 접근 실패 시 조용히 넘어감
        else:
            pass # 검색 결과 없음 시 조용히 넘어감
            
    except Exception as e:
        pass # 각종 에러 발생 시 프로그램 멈추지 않고 계속 진행
        
    time.sleep(0.15) # API 과부하 방지

# 5. 최종 결과 저장
mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv("local_multimodal_mapping.csv", index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print(f"🎉 밸런싱 데이터셋 구축 완료! 총 {success_count}개의 이미지가 확보되었습니다.")
print(f"📄 매칭 데이터 파일: local_multimodal_mapping.csv")
print("="*50)