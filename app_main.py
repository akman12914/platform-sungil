# app_main.py
import streamlit as st
import ui_theme as ui

st.set_page_config(
    page_title="SUNGIL UBR Suite",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.apply()  # 공통 스타일 적용

ui.hero(
    title="SUNGIL UBR Suite",
    subtitle="바닥 · 벽 · 천장 계산과 시방서 QA를 빠르고 정확하게",
)

st.markdown("### 빠른 이동")

# Streamlit 1.30+ 에서 st.page_link 지원
cols = st.columns(4)
with cols[0]:
    st.page_link(
        "pages/01_Floor_UI.py",
        label="🟣 바닥 계산",
        help="규격표/치수 기반 PVE 견적·도식",
    )
with cols[1]:
    st.page_link(
        "pages/02_Wall_UI.py", label="🟢 벽 분할", help="타일 규격 기반 패널 분할"
    )
with cols[2]:
    st.page_link(
        "pages/03_Ceiling_UI.py",
        label="🔵 천장 최적화",
        help="카탈로그 기반 Body/Side 최적화",
    )
with cols[3]:
    st.page_link(
        "pages/04_Chat_Bot.py", label="💬 시방서 QA 챗봇", help="문서 질의응답"
    )

st.markdown("---")

# 참고 / 사용 가이드
with st.container():
    st.subheader("사용 가이드")
    st.markdown(
        """
- 각 페이지는 **완전히 독립적**으로 동작합니다. (자체 업로더/입력/결과)
- 업로더는 **자동 적용 + 초기화 버튼**을 지원해 캐시/포인터 문제 없이 안정 동작합니다.
- 바닥/벽/천장의 계산 로직은 기존 파일(`floor_panel_final.py`, `wall_panel_final.py`, `ceil_panel_final.py`)을 그대로 사용합니다.
    """
    )
    st.info(
        "TIP: 페이지 우측 상단 메뉴 → **Clear cache** 로 데이터가 꼬였을 때 초기화할 수 있어요."
    )
