import streamlit as st

try:
    st.set_page_config(page_title="UBR 통합 런처", layout="wide")
except Exception:
    pass

st.markdown("## UBR 통합 런처")
st.markdown(
    """
**사용방법**  
1) 이 화면에서 원하는 **페이지 버튼**을 클릭하세요. (Multipage 구조)  
2) 각 페이지의 **사이드바**에서 치수/옵션을 설정하세요. 스타일만 정리했고, 입력 로직은 변경하지 않았습니다.  
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/original_floor.py", label="바닥판 계산기로 이동", icon="🟦")
with col2:
    st.page_link("pages/original_wall.py", label="벽판 계산기로 이동", icon="🟩")
with col3:
    st.page_link("pages/original_ceil.py", label="천장판 최적화로 이동", icon="🟨")

st.divider()
st.caption("Multipage 구조로 변경했습니다. 아래 명령으로 실행하세요.")
st.code(
    "streamlit run app_main.py",
    language="bash",
)
