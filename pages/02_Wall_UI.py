import streamlit as st
import pandas as pd
from wall_panel_final import compute_layout, effective_height

st.set_page_config(page_title="UBR · 벽", layout="wide")

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

st.title("벽 분할")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**입력**")
    tile = st.selectbox("타일 규격", ["300×600", "250×400"], key="W_tile")
    TH, TW = (300, 600) if tile == "300×600" else (250, 400)

    base_H = st.number_input("벽 높이 H (mm)", 1500, 4800, 2400, step=50, key="W_H")
    pve_flag = st.toggle("바닥 소재가 PVE (벽 +50mm 보정)", value=True, key="W_pve")
    H_eff = effective_height(base_H, "PVE" if pve_flag else "FRP/GRP")

    W_face = st.number_input("대상 벽 폭 W (mm)", 600, 4800, 2200, step=50, key="W_W")

    st.caption(f"유효 높이 H': **{H_eff} mm**")
    run = st.button("벽 분할 계산", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**결과**")
    if run:
        try:
            panels, rule = compute_layout(W_face, H_eff, TH, TW)
            df = pd.DataFrame(
                [
                    {
                        "종류": p.kind,
                        "위치": p.pos,
                        "W": p.w,
                        "H": p.h,
                        "라벨": p.label(),
                    }
                    for p in panels
                ]
            )
            st.dataframe(df, use_container_width=True)
            st.success(f"규칙: {rule}")
            st.markdown("<div class='kpi'>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='item'><span style='color:#64748b'>패널 수</span><b>{len(panels)}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='item'><span style='color:#64748b'>타일</span><b>{tile}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"분할 실패: {e}")
    else:
        st.info("좌측에서 입력 후 ‘벽 분할 계산’을 누르세요.")
    st.markdown("</div>", unsafe_allow_html=True)
