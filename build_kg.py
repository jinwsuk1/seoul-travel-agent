import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import platform

# 1. 폰트 설정 (한글 깨짐 방지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

print("🧠 지식 그래프(KG)를 구축하고 시각화합니다...")

# 2. 정제된 데이터 불러오기
df = pd.read_csv("refined_seoul_spots.csv", low_memory=False)

# 지역구 추출 로직
def extract_gu(address):
    try:
        parts = str(address).split()
        for part in parts:
            if part.endswith('구') and len(part) < 5:
                return part
        return '기타'
    except:
        return '기타'

df['지역구'] = df['도로명주소'].apply(extract_gu)

# ---------------------------------------------------------
# 💡 메모리 부하 방지 및 발표용 시각화를 위해 
# 특정 지역(예: 강남구, 서초구)의 데이터 40개만 샘플링합니다.
# ---------------------------------------------------------
sample_df = df[df['지역구'].isin(['강남구', '서초구'])].head(40)

# 3. NetworkX 그래프 객체 생성 (방향성이 있는 MultiDiGraph)
G = nx.MultiDiGraph()

# 4. 온톨로지(Ontology) 규칙에 따라 노드와 엣지(관계) 추가
for _, row in sample_df.iterrows():
    restaurant = row['사업장명']
    district = row['지역구']
    category = row['업태구분명']
    
    # 노드 추가 (식당, 지역구, 카테고리)
    G.add_node(restaurant, type='Restaurant')
    G.add_node(district, type='Location')
    G.add_node(category, type='Category')
    
    # 엣지(관계) 추가
    G.add_edge(restaurant, district, relation='LOCATED_IN')
    G.add_edge(restaurant, category, relation='IS_A')

# 5. 그래프 시각화 디자인 설정
plt.figure(figsize=(14, 10))

# 노드 종류별로 색상과 크기 분리
node_colors = []
node_sizes = []
for node, attr in G.nodes(data=True):
    if attr.get('type') == 'Location':
        node_colors.append('#ff9999') # 지역구는 빨간색
        node_sizes.append(3000)
    elif attr.get('type') == 'Category':
        node_colors.append('#66b3ff') # 카테고리는 파란색
        node_sizes.append(2500)
    else:
        node_colors.append('#99ff99') # 식당은 초록색
        node_sizes.append(1000)

# 그래프 레이아웃 배치 (spring_layout이 가장 자연스럽게 펼쳐집니다)
pos = nx.spring_layout(G, k=0.5, seed=42)

# 노드 그리기
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='white')
# 엣지(선) 그리기
nx.draw_networkx_edges(G, pos, edge_color='#cccccc', arrows=True, arrowsize=15)
# 라벨(글자) 그리기
nx.draw_networkx_labels(G, pos, font_family=plt.rcParams['font.family'], font_size=9, font_weight='bold')

# 엣지 라벨 (LOCATED_IN, IS_A) 그리기
edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='red')

plt.title('서울 맛집 지식 그래프 온톨로지 (Knowledge Graph Ontology)', fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()

# 고해상도 이미지로 저장
plt.savefig('4_지식그래프_온톨로지.png', dpi=300)
plt.close()

print("✅ [4_지식그래프_온톨로지.png] 저장 완료! 발표 자료에 추가하세요.")