import streamlit as st

try:
    st.set_page_config(page_title="UBR 통합 런처", layout="wide")
except Exception:
    pass

st.markdown("## UBR 통합 런처")
st.markdown(
    """
**사용방법**  
1) 아래 버튼으로 각 페이지로 이동하거나, `streamlit run <파일>`로 개별 실행합니다.  
2) 각 페이지의 **사이드바**에서 치수/옵션을 설정하세요. (이 파일은 디자인만 정리하며, 사이드바 입력 로직은 수정하지 않았습니다.)  
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("floor_panel_final.py", label="바닥판 계산기로 이동", icon="🟦")
with col2:
    st.page_link("wall_panel_final.py", label="벽판 계산기로 이동", icon="🟩")
with col3:
    st.page_link("ceil_panel_final.py", label="천장판 최적화로 이동", icon="🟨")

st.divider()
st.caption("`st.page_link`가 동작하지 않으면 아래 명령으로 개별 실행하세요.")
st.code(
    "streamlit run floor_panel_final.py\nstreamlit run wall_panel_final.py\nstreamlit run ceil_panel_final.py",
    language="bash",
)
