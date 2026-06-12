import streamlit as st
from travel_agent import run_agentic_rag
import re
import uuid

# ─────────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="서울 맛집 AI",
    page_icon="🍜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 전역 CSS (보조 스타일링 및 프리미엄 디자인 적용)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Noto+Serif+KR:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Noto Sans KR', sans-serif !important;
    background-color: #0b0d14 !important;
}

/* 배경 방사형 글로우 */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 70% 50% at 50% 20%, rgba(30,58,138,0.32) 0%, transparent 65%),
        radial-gradient(ellipse 40% 30% at 75% 75%, rgba(79,70,229,0.10) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* 헤더/푸터 제거 */
header[data-testid="stHeader"], footer { display: none !important; }

/* 메인 너비 */
[data-testid="stMainBlockContainer"] {
    max-width: 780px !important;
    padding: 0 2rem !important;
}

/* 웰컴 영역 */
.welcome-wrap {
    text-align: center;
    padding: 5rem 0 2rem;
}
.welcome-title {
    font-family: 'Noto Serif KR', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #f1f5f9;
    line-height: 1.4;
    margin-bottom: 0.6rem;
}
.welcome-title span {
    background: linear-gradient(90deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.welcome-sub {
    font-size: 1rem;
    color: #475569;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

/* 웰컴 화면 입력 필드 (st.text_input) */
[data-baseweb="input"] {
    border-radius: 30px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 0.2rem 1.4rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-baseweb="input"]:hover {
    border-color: rgba(129, 140, 248, 0.3) !important;
    background: rgba(15, 23, 42, 0.75) !important;
}
[data-baseweb="input"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8, 0 0 20px rgba(129, 140, 248, 0.2) !important;
    background: rgba(15, 23, 42, 0.85) !important;
}
[data-baseweb="input"] input {
    font-size: 1.1rem !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #f8fafc !important;
    height: 52px !important;
    padding: 0 !important;
}
[data-baseweb="input"] input::placeholder {
    color: #64748b !important;
    font-weight: 300 !important;
}

/* 폼 제출 버튼 숨김 */
[data-testid="stFormSubmitButton"] { display: none !important; }

/* 예시 칩 */
.suggestion-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.5rem;
}
.chip {
    padding: 8px 18px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.04);
    color: #94a3b8;
    font-size: 0.88rem;
    font-family: 'Noto Sans KR', sans-serif;
    white-space: nowrap;
}

/* 채팅 메시지 */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 20px !important;
    padding: 1rem 1.3rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
}

/* 채팅창 입력 영역 (st.chat_input) */
[data-testid="stChatInput"] {
    border-radius: 30px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background-color: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stChatInput"]:hover {
    border-color: rgba(129, 140, 248, 0.3) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8, 0 0 20px rgba(129, 140, 248, 0.2) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 1.05rem !important;
    color: #f8fafc !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    line-height: 1.5 !important;
    background-color: transparent !important;
    padding-top: 6px !important;
}
/* 전송 버튼 */
[data-testid="stChatInput"] button {
    background-color: transparent !important;
    border: none !important;
    color: #818cf8 !important;
    transition: transform 0.2s, color 0.2s !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.1) !important;
    color: #a78bfa !important;
}

/* 마크다운 */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    font-size: 1rem !important;
    line-height: 1.85 !important;
}
[data-testid="stChatMessageContent"] h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 1.2rem 0 0.4rem !important;
}
[data-testid="stChatMessageContent"] a {
    color: #818cf8 !important;
    text-decoration: none !important;
    border-bottom: 1px solid rgba(129,140,248,0.3) !important;
}
[data-testid="stChatMessageContent"] img {
    border-radius: 14px !important;
    max-width: 100% !important;
}
[data-testid="stChatMessageContent"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1rem 0 !important;
}

/* 하단 배경 (대화 중) */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background-color: #0b0d14 !important;
    border-top: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"]::before { display: none !important; }

/* 스크롤바 */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 10px; }

/* 초기화 버튼 */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ─────────────────────────────────────────────
 * 캐러셀 / 슬라이드 컴포넌트 스타일
 * ───────────────────────────────────────────── */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ─────────────────────────────────────────────
 * 세로 적층형 카드 목록 스타일
 * ───────────────────────────────────────────── */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(24px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.cards-wrapper {
    display: flex;
    flex-direction: column;
    gap: 24px;
    width: 100%;
    margin: 1.5rem 0;
}

.restaurant-card {
    width: 100%;
    background: rgba(22, 28, 45, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s, border-color 0.3s;
    box-sizing: border-box;
    opacity: 0;
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.restaurant-card:hover {
    border-color: rgba(129, 140, 248, 0.25);
    transform: translateY(-2px);
}

/* 순차적 애니메이션 등장 딜레이 */
.restaurant-card:nth-child(1) { animation-delay: 0.05s; }
.restaurant-card:nth-child(2) { animation-delay: 0.18s; }
.restaurant-card:nth-child(3) { animation-delay: 0.30s; }

.card-image-wrap {
    width: 100%;
    height: 240px;
    position: relative;
    overflow: hidden;
}
.card-image-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
}
.restaurant-card:hover .card-image-wrap img {
    transform: scale(1.03);
}
.card-image-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60%;
    background: linear-gradient(to top, rgba(11, 13, 20, 0.95) 0%, transparent 100%);
    z-index: 1;
}
.card-title-overlay {
    position: absolute;
    bottom: 20px;
    left: 24px;
    right: 24px;
    font-family: 'Noto Serif KR', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    margin: 0 !important;
    z-index: 2;
}

.card-body {
    padding: 24px;
}

.card-meta-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
}
.card-meta-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.4;
}
.card-meta-item svg {
    width: 16px;
    height: 16px;
    margin-top: 2px;
    flex-shrink: 0;
    fill: currentColor;
}

.card-description {
    color: #cbd5e1;
    font-size: 0.98rem;
    line-height: 1.75;
    margin-bottom: 20px !important;
    font-weight: 300;
}

.card-recommendation {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-left: 4px solid #818cf8;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 20px;
}
.card-rec-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #a5b4fc;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-rec-text {
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin: 0;
}

.card-actions {
    display: flex;
    gap: 12px;
}
.card-btn {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 0.88rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: all 0.2s;
    box-sizing: border-box;
}
.card-btn svg {
    flex-shrink: 0;
}
.card-btn-kakao {
    background: rgba(129, 140, 248, 0.1);
    color: #a5b4fc !important;
    border: 1px solid rgba(129, 140, 248, 0.2);
}
.card-btn-kakao:hover {
    background: #818cf8;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(129, 140, 248, 0.3);
}
.card-btn-naver {
    background: rgba(167, 139, 250, 0.1);
    color: #c084fc !important;
    border: 1px solid rgba(167, 139, 250, 0.2);
}
.card-btn-naver:hover {
    background: #a78bfa;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(167, 139, 250, 0.3);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 캐러셀 / 슬라이드 파싱 및 렌더링 헬퍼 함수
# ─────────────────────────────────────────────
def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.splitlines())


def parse_restaurants(text):
    if "### 🍽️" not in text:
        return []
    
    blocks = text.split("\n---\n")
    restaurants = []
    
    for block in blocks:
        block = block.strip()
        if not block or "### 🍽️" not in block:
            continue
        
        name = ""
        address = ""
        phone = ""
        summary = ""
        kg_rec = ""
        img_url = ""
        kakao_url = ""
        naver_url = ""
        
        name_match = re.search(r"### 🍽️\s*(.*)", block)
        if name_match:
            name = name_match.group(1).strip()
            
        addr_match = re.search(r"-\s*\*\*주소\*\*:\s*(.*)", block)
        if addr_match:
            address = addr_match.group(1).strip()
            
        phone_match = re.search(r"-\s*\*\*전화번호\*\*:\s*(.*)", block)
        if phone_match:
            phone = phone_match.group(1).strip()
            
        summary_match = re.search(r"-\s*\*\*위치 및 특징\*\*:\s*(.*)", block)
        if summary_match:
            summary = summary_match.group(1).strip()
            
        kg_match = re.search(r"-\s*\*\*💡\s*GraphRAG 추천\*\*:\s*(.*)", block)
        if kg_match:
            kg_rec = kg_match.group(1).strip()
            
        img_match = re.search(r"!\[식당 사진\]\((.*?)\)", block)
        if img_match:
            img_url = img_match.group(1).strip()
            
        kakao_match = re.search(r"\*\s*🗺️\s*\[카카오맵 바로가기\]\((.*?)\)", block)
        if kakao_match:
            kakao_url = kakao_match.group(1).strip()
            
        naver_match = re.search(r"\*\s*🔍\s*\[네이버 지도 바로가기\]\((.*?)\)", block)
        if naver_match:
            naver_url = naver_match.group(1).strip()
            
        if name:
            restaurants.append({
                "name": name,
                "address": address or "주소 정보 없음",
                "phone": phone,
                "summary": summary,
                "kg_rec": kg_rec,
                "img_url": img_url or "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&q=80&w=600",
                "kakao_url": kakao_url,
                "naver_url": naver_url
            })
    return restaurants


def render_restaurant_cards(restaurants):
    if not restaurants:
        return ""
    
    cards_html = ""
    for i, rest in enumerate(restaurants):
        phone_html = ""
        if rest["phone"]:
            phone_html = f"""
            <div class="card-meta-item">
                <svg viewBox="0 0 24 24"><path d="M6.62,10.79C8.06,13.62 10.38,15.94 13.21,17.38L15.41,15.18C15.69,14.9 16.08,14.82 16.43,14.93C17.55,15.3 18.75,15.5 20,15.5A1,1 0 0,1 21,16.5V20A1,1 0 0,1 20,21A17,17 0 0,1 3,4A1,1 0 0,1 4,3H7.5A1,1 0 0,1 8.5,4C8.5,5.25 8.7,6.45 9.07,7.57C9.18,7.92 9.1,8.31 8.82,8.59L6.62,10.79Z"/></svg>
                <span>{rest['phone']}</span>
            </div>
            """
            
        rec_html = ""
        if rest["kg_rec"]:
            rec_html = f"""
            <div class="card-recommendation">
                <div class="card-rec-title">
                    <svg viewBox="0 0 24 24" style="width: 14px; height: 14px; fill: #818cf8;"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M12,6A1.5,1.5 0 0,0 10.5,7.5A1.5,1.5 0 0,0 12,9A1.5,1.5 0 0,0 13.5,7.5A1.5,1.5 0 0,0 12,6M11,10V18H13V10H11Z"/></svg>
                    <span>GraphRAG 추천 대안</span>
                </div>
                <p class="card-rec-text">{rest['kg_rec']}</p>
            </div>
            """
            
        kakao_btn = ""
        if rest["kakao_url"]:
            kakao_btn = f"""
            <a href="{rest['kakao_url']}" target="_blank" class="card-btn card-btn-kakao">
                <svg viewBox="0 0 24 24" style="width: 16px; height: 16px; fill: currentColor;"><path d="M12,2C8.13,2 5,5.13 5,9C5,14.25 12,22 12,22C12,22 19,14.25 19,9C19,5.13 15.87,2 12,2M12,11.5A2.5,2.5 0 0,1 9.5,9A2.5,2.5 0 0,1 12,6.5A2.5,2.5 0 0,1 14.5,9A2.5,2.5 0 0,1 12,11.5Z"/></svg>
                카카오맵 바로가기
            </a>
            """
            
        naver_btn = ""
        if rest["naver_url"]:
            naver_btn = f"""
            <a href="{rest['naver_url']}" target="_blank" class="card-btn card-btn-naver">
                <svg viewBox="0 0 24 24" style="width: 16px; height: 16px; fill: currentColor;"><path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z"/></svg>
                네이버 지도 바로가기
            </a>
            """
            
        cards_html += f"""
        <div class="restaurant-card">
            <div class="card-image-wrap">
                <img src="{rest['img_url']}" alt="{rest['name']}">
                <div class="card-image-overlay"></div>
                <h3 class="card-title-overlay">{rest['name']}</h3>
            </div>
            <div class="card-body">
                <div class="card-meta-row">
                    <div class="card-meta-item">
                        <svg viewBox="0 0 24 24"><path d="M12,2C8.13,2 5,5.13 5,9C5,14.25 12,22 12,22C12,22 19,14.25 19,9C19,5.13 15.87,2 12,2M12,11.5A2.5,2.5 0 0,1 9.5,9A2.5,2.5 0 0,1 12,6.5A2.5,2.5 0 0,1 14.5,9A2.5,2.5 0 0,1 12,11.5Z"/></svg>
                        <span>{rest['address']}</span>
                    </div>
                    {phone_html}
                </div>
                <p class="card-description">{rest['summary']}</p>
                {rec_html}
                <div class="card-actions">
                    {kakao_btn}
                    {naver_btn}
                </div>
            </div>
        </div>
        """
        
    html = f"""
    <div class="cards-wrapper">
        {cards_html}
    </div>
    """
    return clean_html(html)


# ─────────────────────────────────────────────
# 세션 초기화
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# [ 상태 A ] 첫 화면 — 웰컴 + 중앙 입력
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-title">서울의 맛,<br><span>어디서 찾으세요?</span></div>
        <div class="welcome-sub">GraphRAG 기반 AI가 36,000여 곳의 맛집 중 최적의 곳을 추천합니다</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("search_form", clear_on_submit=True):
        user_query = st.text_input(
            label="",
            placeholder="어느 지역, 어떤 음식이 드시고 싶으세요?",
            label_visibility="collapsed",
        )
        st.form_submit_button("검색")   # hidden via CSS, Enter키로 제출

    st.markdown("""
    <div class="suggestion-row">
        <span class="chip">🍝 강남 파스타 맛집</span>
        <span class="chip">☕ 홍대 분위기 좋은 카페</span>
        <span class="chip">🍖 마포구 한식 맛집</span>
        <span class="chip">🍜 서울 3대 냉면집</span>
        <span class="chip">🍣 이태원 일식당</span>
    </div>
    """, unsafe_allow_html=True)

    if user_query and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})
        st.rerun()

# ─────────────────────────────────────────────
# [ 상태 B ] 대화 중 — 채팅 UI
# ─────────────────────────────────────────────
else:
    # 초기화 버튼
    cols = st.columns([10, 1])
    with cols[1]:
        if st.button("↺"):
            st.session_state.messages = []
            st.rerun()

    # 첫 질문에 대한 AI 응답이 아직 없을 때
    if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "user":
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[0]["content"])
        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    answer = run_agentic_rag(st.session_state.messages[0]["content"])
                    restaurants = parse_restaurants(answer)
                    if restaurants:
                        cards_html = render_restaurant_cards(restaurants)
                        st.markdown(cards_html, unsafe_allow_html=True)
                    else:
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"오류가 발생했습니다: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
        st.rerun()

    # 대화 기록 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                restaurants = parse_restaurants(msg["content"])
                if restaurants:
                    cards_html = render_restaurant_cards(restaurants)
                    st.markdown(cards_html, unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    # 하단 입력
    if user_query := st.chat_input("계속 물어보세요..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    answer = run_agentic_rag(user_query)
                    restaurants = parse_restaurants(answer)
                    if restaurants:
                        cards_html = render_restaurant_cards(restaurants)
                        st.markdown(cards_html, unsafe_allow_html=True)
                    else:
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"오류가 발생했습니다: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})