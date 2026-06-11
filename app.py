import streamlit as st
# 기존에 작성하신 파일(travel_agent)에서 메인 구동 함수를 불러옵니다.
from travel_agent import run_agentic_rag

# 1. 웹 페이지 기본 설정 (타이틀, 아이콘 등)
st.set_page_config(page_title="서울 맛집 추천 AI 에이전트", page_icon="🍽️", layout="centered")

st.title("🍽️ 서울 맛집 추천 에이전트")
st.caption("GraphRAG와 로컬 멀티모달 기술 기반으로 실시간 맛집을 추천하고 이미지를 렌더링합니다.")

# 2. 세션 상태(Session State) 초기화 (대화 기록 저장용)
# 스트림릿은 입력이 들어올 때마다 코드를 처음부터 다시 실행하므로, 대화 기록을 세션에 보관해야 합니다.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 서울의 어느 지역, 어떤 종류의 맛집을 찾으시나요? (예: 강남역 주변 맛있는 파스타집 추천해줘)"}
    ]

# 3. 기존 대화 기록을 화면에 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자로부터 채팅 입력 받기
if user_query := st.chat_input("맛집을 검색해보세요..."):
    
    # 사용자가 입력한 메시지를 화면에 표시하고 대화 기록에 저장
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 5. AI 에이전트 구동 및 답변 생성 (로딩 스피너 표시)
    with st.chat_message("assistant"):
        with st.spinner("🔍 에이전트가 데이터베이스와 실시간 웹을 검색하고 있습니다..."):
            try:
                # 기존 travel_agent.py의 오케스트레이션 함수 호출
                final_answer = run_agentic_rag(user_query)
                
                # 결과 출력 (이때 마크다운 안의 ![식당 사진](Base64데이터)가 이미지로 자동 변환됩니다!)
                st.markdown(final_answer)
                
                # 대화 기록에 백업
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                error_msg = f"❌ 에러가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})