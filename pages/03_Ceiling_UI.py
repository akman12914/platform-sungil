import io
import streamlit as st
import pandas as pd
from ceil_panel_final import (
    parse_catalog,
    sample_catalog,
    optimize_rect,
    optimize_corner,
)

st.set_page_config(page_title="UBR · 천장", layout="wide")

st.markdown(
    """
<style>
.card{background:#f6f7fb;border:1px solid #e5e7eb;border-radius:16px;padding:18px;margin-bottom:14px;}
.kpi{display:flex;gap:16px;flex-wrap:wrap;}
.kpi .item{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;min-width:180px}
.kpi .item b{display:block;font-size:22px;line-height:1.2}
</style>
""",
    unsafe_allow_html=True,
)

# 세션
for k, v in {"C_bytes": None, "C_name": None, "C_counter": 0}.items():
    st.session_state.setdefault(k, v)

st.title("천장 최적화")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "**카탈로그 업로드**  <span style='color:#64748b'>(시트명: '천창판' / '천장판')</span>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader(
        "업로드 (.xlsx)", type=["xlsx"], key=f"C_up_{st.session_state['C_counter']}"
    )
    if up is not None:
        st.session_state["C_bytes"] = up.getvalue()
        st.session_state["C_name"] = up.name
        st.success(f"적용됨: {up.name}")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption(f"현재 적용: **{st.session_state['C_name'] or '없음'}**")
    with c2:
        if st.button("초기화", key="C_reset"):
            st.session_state["C_bytes"] = None
            st.session_state["C_name"] = None
            st.session_state["C_counter"] += 1
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**치수 입력(직사각 기준)**")
    Wc = st.number_input("천장 폭 W (mm)", 600, 1900, 1300, step=10, key="C_W")
    Lc = st.number_input("천장 길이 L (mm)", 800, 4000, 1750, step=10, key="C_L")
    cut_cost = st.number_input("컷 비용(원/컷)", 0, 100000, 3000, step=500, key="C_cut")

    run_rect = st.button(
        "천장 최적화 (직사각)", type="primary", use_container_width=True
    )

    with st.expander("코너형(세면/샤워 분리)"):
        S_W = st.number_input("세면부 폭", 600, 1900, 1300, step=10, key="C_S_W")
        S_L = st.number_input("세면부 길이", 800, 4000, 1750, step=10, key="C_S_L")
        H_W = st.number_input("샤워부 폭", 400, 1900, 900, step=10, key="C_H_W")
        H_L = st.number_input("샤워부 길이", 400, 4000, 900, step=10, key="C_H_L")
        run_corner = st.button(
            "천장 최적화 (코너형)", use_container_width=True, key="C_btn_corner"
        )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**카탈로그 미리보기 & 결과**")

    # 카탈로그 로딩
    if st.session_state["C_bytes"]:
        try:
            bio = io.BytesIO(st.session_state["C_bytes"])
            df_check, df_body, df_side = parse_catalog(bio)
        except Exception:
            st.warning("시트 자동 인식 실패 → 샘플 카탈로그 사용")
            df_check, df_body, df_side = sample_catalog()
    else:
        df_check, df_body, df_side = sample_catalog()

    with st.expander("점검구 / 바디 / 사이드 미리보기", expanded=False):
        st.dataframe(df_check.head(8), use_container_width=True)
        st.dataframe(df_body.head(8), use_container_width=True)
        st.dataframe(df_side.head(8), use_container_width=True)

    # 최적화 실행
    if run_rect:
        res = optimize_rect(Wc, Lc, df_check, df_body, df_side, cut_cost, 25.0)
        st.write(res)
    elif "C_btn_corner" in st.session_state and st.session_state["C_btn_corner"]:
        res = optimize_corner(
            S_W, S_L, H_W, H_L, df_check, df_body, df_side, cut_cost, 25.0
        )
        st.write(res)
    else:
        st.info("좌측에서 업로드/치수 입력 후 버튼을 눌러주세요.")

    st.markdown("</div>", unsafe_allow_html=True)
