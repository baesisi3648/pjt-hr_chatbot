import streamlit as st
from graph import app as workflow # 우리가 만든 그래프 엔진 가져오기

# 1. 페이지 설정
st.set_page_config(page_title="ZIC-TALK", page_icon="🤖")
st.title("🤖 ZIC-TALK HR 챗봇")
st.caption("🚀 LangGraph 기반 3중 검증(Draft-Critic-Rewrite) 엔진 탑재")

# 2. 대화 기록 초기화 (새로고침 해도 대화 유지)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 취업규칙에 대해 무엇이든 물어보세요. (예: 연차는 얼마나 주나요?)"}]

# 3. 이전 대화 내용 화면에 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # (1) 사용자 메시지 화면 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # (2) AI 답변 생성 (Spinner로 대기 표시)
    with st.spinner("규정 검색 및 팩트체크 중입니다... (약 10초 소요)"):
        try:
            # LangGraph 엔진 실행!
            inputs = {"question": prompt}
            
            # invoke로 최종 결과만 받아오기
            # (중간 과정을 보고 싶으면 stream을 써야 하지만, MVP는 심플하게!)
            result = workflow.invoke(inputs)
            
            final_answer = result["draft"] # 최종 수정된 답변은 'draft' 키에 저장됨
            
            # (3) AI 응답 화면 표시
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.chat_message("assistant").write(final_answer)
            
            # (선택) 디버깅용: 몇 번 수정했는지 보기
            if result.get("revision_count", 0) > 0:
                st.info(f"💡 정확도를 위해 답변을 {result['revision_count']}회 수정하여 제공했습니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")