import io
import streamlit as st
import pandas as pd
from floor_panel_final import pve_quote, draw_bathroom

st.set_page_config(page_title="UBR · 바닥", layout="wide")

# ---------- 스타일 ----------
st.markdown(
    """
<style>
:root{--brand:#4F46E5;--muted:#f6f7fb;--text:#0f172a;}
.block-container{padding-top:2rem;}
.card{background:var(--muted);border:1px solid #e5e7eb;border-radius:16px;padding:18px;margin-bottom:14px;}
.hr{height:1px;background:#e5e7eb;margin:12px 0;}
.kpi{display:flex;gap:16px;flex-wrap:wrap;}
.kpi .item{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;min-width:180px}
.kpi .item b{display:block;font-size:22px;line-height:1.2}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- 세션 키 ----------
for k, v in {"F_bytes": None, "F_name": None, "F_counter": 0}.items():
    st.session_state.setdefault(k, v)

st.title("바닥 계산")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**입력**")
    c1, c2 = st.columns(2)
    with c1:
        shape = st.radio("형태", ["사각형", "코너형"], horizontal=True, key="F_shape")
        central = st.radio("중앙배수", ["No", "Yes"], horizontal=True, key="F_central")
        btype = st.radio("유형", ["샤워형", "욕조형", "구분없음"], key="F_type")
    with c2:
        units = st.number_input("공사 세대수", 1, 100000, 100, key="F_units")
        mgmt = st.number_input("관리비율(%)", 0.0, 100.0, 25.0, step=0.5, key="F_mgmt")

    col1, col2 = st.columns(2)
    with col1:
        bw = st.number_input("욕실 폭 BW (mm)", 400, 6000, 1500, step=10, key="F_bw")
        sw = st.number_input(
            "세면부 폭 (mm)",
            0,
            6000,
            1300,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음" or shape == "코너형"),
            key="F_sw",
        )
        shw = st.number_input(
            "샤워부 폭 (mm)",
            0,
            6000,
            800,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음" or shape == "코너형"),
            key="F_shw",
        )
    with col2:
        bl = st.number_input("욕실 길이 BL (mm)", 400, 6000, 2200, step=10, key="F_bl")
        sl = st.number_input(
            "세면부 길이 (mm)",
            0,
            6000,
            1500,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음" or shape == "코너형"),
            key="F_sl",
        )
        shl = st.number_input(
            "샤워부 길이 (mm)",
            0,
            6000,
            900,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음" or shape == "코너형"),
            key="F_shl",
        )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        "**바닥판 규격 엑셀 (선택)**  <span style='color:#64748b'>(시트명 '바닥판' 권장)</span>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader(
        "업로드 (.xlsx)", type=["xlsx"], key=f"F_up_{st.session_state['F_counter']}"
    )
    if up is not None:
        st.session_state["F_bytes"] = up.getvalue()
        st.session_state["F_name"] = up.name
        st.success(f"적용됨: {up.name}")
    colx, coly = st.columns([1, 1])
    with colx:
        st.caption(f"현재 적용: **{st.session_state['F_name'] or '없음'}**")
    with coly:
        if st.button("초기화", key="F_reset"):
            st.session_state["F_bytes"] = None
            st.session_state["F_name"] = None
            st.session_state["F_counter"] += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("바닥 계산", type="primary", use_container_width=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**결과**")

    if run:
        pve_kind = st.selectbox(
            "PVE 유형", ["일반형(+380mm)", "주거약자(+480mm)"], key="F_pve_kind"
        )
        kind = "일반형" if pve_kind.startswith("일반") else "주거약자"
        pve = pve_quote(bw, bl, mgmt / 100.0, kind=kind)

        if st.session_state["F_bytes"]:
            try:
                bio = io.BytesIO(st.session_state["F_bytes"])
                try:
                    df = pd.read_excel(bio, sheet_name="바닥판", engine="openpyxl")
                except Exception:
                    bio.seek(0)
                    df = pd.read_excel(bio, engine="openpyxl")  # 첫 시트
                st.dataframe(df.head(12), use_container_width=True)
            except Exception as e:
                st.warning(f"바닥 규격표 읽기 실패: {e}")

        try:
            img = draw_bathroom(
                shape,
                bw,
                bl,
                (
                    None
                    if (central == "Yes" or btype == "구분없음" or shape == "코너형")
                    else sw
                ),
                (
                    None
                    if (central == "Yes" or btype == "구분없음" or shape == "코너형")
                    else sl
                ),
                (
                    None
                    if (central == "Yes" or btype == "구분없음" or shape == "코너형")
                    else shw
                ),
                (
                    None
                    if (central == "Yes" or btype == "구분없음" or shape == "코너형")
                    else shl
                ),
                central=central,
                btype=btype,
            )
            st.image(img, caption="개략 도식")
        except Exception:
            pass

        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        total = int(pve.get("total", 0)) if isinstance(pve, dict) else 0
        st.markdown(
            f"<div class='item'><span style='color:#64748b'>PVE 견적</span><b>{total:,} 원</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='item'><span style='color:#64748b'>세대수</span><b>{units}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("좌측에서 입력/업로드 후 ‘바닥 계산’을 누르세요.")

    st.markdown("</div>", unsafe_allow_html=True)
