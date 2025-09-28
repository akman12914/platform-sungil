import streamlit as st
import pandas as pd

# 코어 함수 import (파일명은 업로드된 그대로 사용)
from floor_panel_final import (
    pve_quote,                     # PVE 견적 계산
    match_center_drain,            # 중앙배수 매칭
    match_non_center_rectangle,    # 사각형(비중앙) 매칭
    match_corner_shower,           # 코너형(비중앙, 샤워형) 매칭
    draw_bathroom                  # 도식 렌더(선택)
)
from wall_panel_final import (
    compute_layout,                # (W,H,TH,TW) -> 패널 분할
    effective_height               # 바닥이 PVE면 +50
)
from ceil_panel_final import (
    parse_catalog,                 # 천장 카탈로그 파서
    sample_catalog,                # 샘플 카탈로그(없을 때)
    optimize_rect,                 # 직사각 최적화
    optimize_corner                # 코너형 최적화
)

st.set_page_config(page_title="UBR 통합 견적 도우미", layout="wide")
st.title("UBR 통합: 바닥 → 벽 → 천장")

# =============== 공통 입력 ===============
with st.sidebar:
    st.header("공통 입력")
    units = st.number_input("공사 세대수", min_value=1, value=100)
    shape  = st.radio("욕실 형태", ["사각형", "코너형"], horizontal=True)
    central = st.radio("중앙배수", ["No", "Yes"], horizontal=True)
    btype = st.radio("욕실 유형", ["샤워형", "욕조형", "구분없음"], horizontal=True)

    st.subheader("욕실 치수 (mm)")
    bw = st.number_input("욕실 폭(BW)", 400, 5000, 1500, step=10)
    bl = st.number_input("욕실 길이(BL)", 400, 6000, 2200, step=10)

    if shape == "사각형":
        sw = st.number_input("세면부 폭", 0, 5000, 1300, step=10, disabled=(central=="Yes" or btype=="구분없음"))
        sl = st.number_input("세면부 길이",0, 6000, 1500, step=10, disabled=(central=="Yes" or btype=="구분없음"))
        shw= st.number_input("샤워부 폭", 0, 5000, 800,  step=10, disabled=(central=="Yes" or btype=="구분없음"))
        shl = st.number_input("샤워부 길이",0, 6000, 900,  step=10, disabled=(central=="Yes" or btype=="구분없음"))
    else:
        st.caption("코너형일 경우 벽폭 분해치/영역 치수는 벽·천장 탭에서 세부 입력")

    st.divider()
    mgmt_rate_pct = st.number_input("관리비율(%)", 0.0, 100.0, 25.0, step=0.5)
    pve_kind = st.radio("PVE 유형", ["일반형(+380mm)", "주거약자(+480mm)"], horizontal=False)

# 세션 기본값
st.session_state.setdefault("floor_meta", {})
st.session_state.setdefault("wall_result", {})
st.session_state.setdefault("ceil_result", {})

tab_floor, tab_wall, tab_ceil, tab_sum = st.tabs(["바닥", "벽", "천장", "Summary"])

# =============== 1) 바닥 ===============
with tab_floor:
    st.subheader("바닥 계산")

    # 엑셀 테이블(바닥판 규격표)이 있다면 DataFrame 업로드/정규화해서 매칭…(생략 가능)
    st.write("조건에 맞는 바닥판 자동 매칭 및 PVE 견적 산출")

    match_result = None
    if central == "Yes":
        # 중앙배수: GRP(중앙배수 계열) 매칭
        st.caption("중앙배수 Yes → GRP(중앙배수) 우선 매칭")
        # 실제로는 규격표 DataFrame을 넣어야 함. 여기서는 외부 테이블 없이 흐름만 유지.
        # match_center_drain(df, shape, btype, bw, bl)
    else:
        if shape == "사각형":
            # 종류별 매칭
            # match_non_center_rectangle(df, btype, bw, bl, sw, sl, shw, shl)
            pass
        else:
            # match_corner_shower(df, bw, bl, sw, sl, shw, shl)
            pass

    # PVE 견적(선택적): 바닥 소재가 PVE로 결정될 때
    floor_is_pve = st.toggle("바닥 소재: PVE로 가정", value=True)
    if floor_is_pve:
        kind = "일반형" if "일반" in pve_kind else "주거약자"
        pve = pve_quote(bw, bl, mgmt_rate_pct/100.0, kind=kind)  # 원재료비/가공비/관리비 포함 소계
        st.write("PVE 견적:", pve)
        st.session_state["floor_meta"] = {"type": "PVE", "pve_detail": pve, "kind": kind}
    else:
        st.session_state["floor_meta"] = {"type": "FRP/GRP"}

    # 미니 렌더(선택)
    try:
        img = draw_bathroom(shape, bw, bl,
                            None if (central=="Yes" or btype=="구분없음") else sw,
                            None if (central=="Yes" or btype=="구분없음") else sl,
                            None if (central=="Yes" or btype=="구분없음") else shw,
                            None if (central=="Yes" or btype=="구분없음") else shl,
                            central=central, btype=btype)
        st.image(img, caption="개략 도식")
    except Exception:
        pass

    st.success("바닥 단계 완료 → 다음 탭으로 진행하세요.")

# =============== 2) 벽 ===============
with tab_wall:
    st.subheader("벽판 분할(타일 규격 기반)")
    tile = st.selectbox("타일 규격", ["300×600", "250×400"])
    TH, TW = (300, 600) if tile == "300×600" else (250, 400)

    # 바닥 유형에 따른 유효 높이 보정
    base_H = st.number_input("벽 높이(H, mm)", 1500, 4800, 2400, step=50)
    floor_type = st.session_state["floor_meta"].get("type", "FRP/GRP")
    H_eff = effective_height(base_H, floor_type)  # PVE면 +50, 아니면 그대로
    st.write(f"유효 높이(H'): {H_eff} mm  (바닥:{floor_type})")

    # 폭은 벽면 폭(한 면 기준). 데모: 욕실 길이를 상벽(W1=BL)로 간주
    W_face = st.number_input("대상 벽 폭(W, mm)", 600, 4800, bl, step=50)
    if st.button("벽 분할 계산"):
        try:
            panels, rule_label = compute_layout(W_face, H_eff, TH, TW)
            df = pd.DataFrame([{"종류":p.kind, "위치":p.pos, "W":p.w, "H":p.h, "라벨":p.label()} for p in panels])
            st.dataframe(df, use_container_width=True)
            st.info(f"규칙: {rule_label}")
            st.session_state["wall_result"] = {"panels": panels, "rule": rule_label, "tile": tile}
        except Exception as e:
            st.error(f"분할 실패: {e}")

# =============== 3) 천장 ===============
with tab_ceil:
    st.subheader("천장 최적화(카탈로그 기반)")
    up = st.file_uploader("천장 카탈로그 엑셀 업로드(시트명: 천창판/천장판)", type=["xlsx"])

    if up:
        try:
            df_check, df_body, df_side = parse_catalog(up)
        except Exception:
            st.warning("시트 자동 인식 실패 → 샘플 카탈로그 사용")
            df_check, df_body, df_side = sample_catalog()
    else:
        df_check, df_body, df_side = sample_catalog()

    st.write("점검구/바디/사이드 프리뷰")
    st.dataframe(df_check.head(8))
    st.dataframe(df_body.head(8))
    st.dataframe(df_side.head(8))

    st.divider()
    st.markdown("**치수 입력(직사각 기준)**")
    Wc = st.number_input("천장 폭 W (mm)", 600, 1900, 1300, step=10)
    Lc = st.number_input("천장 길이 L (mm)", 800, 4000, 1750, step=10)
    cut_cost = st.number_input("컷 비용(원/컷)", 0, 100000, 3000, step=500)

    if st.button("천장 최적화(직사각)"):
        result = optimize_rect(Wc, Lc, df_check, df_body, df_side, cut_cost, mgmt_rate_pct)
        st.write(result)
        st.session_state["ceil_result"] = result

    with st.expander("코너형(세면부/샤워부 분리)"):
        S_W = st.number_input("세면부 폭", 600, 1900, 1300, step=10, key="s_w")
        S_L = st.number_input("세면부 길이", 800, 4000, 1750, step=10, key="s_l")
        H_W = st.number_input("샤워부 폭",  400, 1900, 900,  step=10, key="h_w")
        H_L = st.number_input("샤워부 길이", 400, 4000, 900,  step=10, key="h_l")

        if st.button("천장 최적화(코너형)"):
            result2 = optimize_corner(S_W, S_L, H_W, H_L, df_check, df_body, df_side, cut_cost, mgmt_rate_pct)
            st.write(result2)
            st.session_state["ceil_result"] = result2

# =============== 4) Summary ===============
with tab_sum:
    st.subheader("요약 / 내보내기")
    floor_meta = st.session_state["floor_meta"]
    wall_result = st.session_state["wall_result"]
    ceil_result = st.session_state["ceil_result"]

    st.markdown("### 바닥")
    st.json(floor_meta)

    st.markdown("### 벽")
    st.json({"tile": wall_result.get("tile"), "rule": wall_result.get("rule"),
             "panels_count": len(wall_result.get("panels", []))})

    st.markdown("### 천장")
    st.json(ceil_result if ceil_result else {"status":"no-run"})

    st.info("필요 시 CSV/엑셀로 내보내기 버튼 추가 가능")

