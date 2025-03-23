# app.py

import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF  # RAG 기반 PDF 질문응답 클래스

# 페이지 기본 설정
st.set_page_config(page_title="RAG with Local DeepSeek R1")


def display_messages():
    """채팅 내역을 화면에 출력합니다."""
    st.subheader("채팅 기록")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))  # 사용자/AI 메시지를 구분하여 출력
    st.session_state["thinking_spinner"] = st.empty()  # 로딩 스피너 공간 초기화


def process_input():
    """사용자 입력을 처리하고 AI 응답을 생성합니다."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()  # 입력 정리
        with st.session_state["thinking_spinner"], st.spinner("생각 중..."):
            try:
                # 사용자 질문에 대한 응답 생성
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        # 채팅 기록에 사용자 질문과 AI 응답 추가
        st.session_state["messages"].append((user_text, True))   # 사용자 메시지
        st.session_state["messages"].append((agent_text, False)) # AI 메시지


def read_and_save_file():
    """업로드된 파일을 읽고 RAG 시스템에 인제스트합니다."""
    st.session_state["assistant"].clear()           # 기존 벡터스토어 초기화
    st.session_state["messages"] = []               # 채팅 기록 초기화
    st.session_state["user_input"] = ""             # 입력창 초기화

    for file in st.session_state["file_uploader"]:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # 인제스트 및 소요 시간 측정
        with st.session_state["ingestion_spinner"], st.spinner(f"{file.name} 문서 인제스트 중..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        # 인제스트 완료 메시지 추가
        st.session_state["messages"].append(
            (f"{file.name} 문서 인제스트 완료 (소요 시간: {t1 - t0:.2f}초)", False)
        )
        os.remove(file_path)  # 임시 파일 삭제


def page():
    """Streamlit 메인 페이지 구성"""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []            # 채팅 내역 저장 공간 초기화
        st.session_state["assistant"] = ChatPDF()    # RAG 시스템 객체 생성

    st.header("로컬 DeepSeek R1 기반 문서 질의응답")

    # 문서 업로드 섹션
    st.subheader("문서 업로드")
    st.file_uploader(
        "PDF 문서를 업로드하세요",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,        # 업로드 시 자동 인제스트
        label_visibility="collapsed",
        accept_multiple_files=True,          # 여러 파일 업로드 가능
    )

    st.session_state["ingestion_spinner"] = st.empty()  # 인제스트 로딩용 공간

    # 검색 관련 설정
    st.subheader("설정")
    st.session_state["retrieval_k"] = st.slider(
        "검색할 문서 조각 수 (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "유사도 점수 기준", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # 채팅창 및 입력창 표시
    display_messages()
    st.text_input("질문을 입력하세요", key="user_input", on_change=process_input)

    # 채팅 기록 초기화 버튼
    if st.button("채팅 초기화"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


# 앱 실행
if __name__ == "__main__":
    page()
