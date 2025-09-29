# app_main.py
import io
import streamlit as st
import pandas as pd

# ===== 코어 함수 import (원본 모듈에서 계산용 함수만 사용) =====
from floor_panel_final import (
    pve_quote,  # PVE 견적 계산
    match_center_drain,  # 중앙배수 매칭 (규격표 DF 필요)
    match_non_center_rectangle,  # 사각형(비중앙) 매칭 (규격표 DF 필요)
    match_corner_shower,  # 코너형(비중앙, 샤워형) 매칭 (규격표 DF 필요)
    draw_bathroom,  # 도식 렌더
)
from wall_panel_final import (
    compute_layout,  # (W,H,TH,TW) -> 패널 분할
    effective_height,  # 바닥이 PVE면 +50
)
from ceil_panel_final import (
    parse_catalog,  # 천장 카탈로그 파서
    sample_catalog,  # 샘플 카탈로그(없을 때)
    optimize_rect,  # 직사각 최적화
    optimize_corner,  # 코너형 최적화
)

# ================== 기본 설정/세션 상태 ==================
st.set_page_config(page_title="UBR 통합 견적 도우미", layout="wide")

# 공통 상태
st.session_state.setdefault("floor_meta", {})
st.session_state.setdefault("wall_result", {})
st.session_state.setdefault("ceil_result", {})

# 업로드 상태 (바닥/천장 각각)
for k, v in {
    "floor_upload_counter": 0,
    "floor_catalog_bytes": None,
    "floor_catalog_name": None,
    "ceil_upload_counter": 0,
    "ceil_catalog_bytes": None,
    "ceil_catalog_name": None,
}.items():
    st.session_state.setdefault(k, v)

st.title("UBR 통합: 바닥 → 벽 → 천장")

# ================== 사이드바 ==================
with st.sidebar:
    st.header("공통 입력")

    units = st.number_input("공사 세대수", min_value=1, value=100, key="units")
    shape = st.radio(
        "욕실 형태", ["사각형", "코너형"], horizontal=True, key="shape_radio"
    )
    central = st.radio("중앙배수", ["No", "Yes"], horizontal=True, key="central_radio")
    btype = st.radio(
        "욕실 유형",
        ["샤워형", "욕조형", "구분없음"],
        horizontal=True,
        key="btype_radio",
    )

    st.subheader("욕실 치수 (mm)")
    bw = st.number_input("욕실 폭(BW)", 400, 5000, 1500, step=10, key="bw")
    bl = st.number_input("욕실 길이(BL)", 400, 6000, 2200, step=10, key="bl")

    # 코너형 대비 기본 None 방지
    sw = st.session_state.get("sw_main", None)
    sl = st.session_state.get("sl_main", None)
    shw = st.session_state.get("shw_main", None)
    shl = st.session_state.get("shl_main", None)

    if shape == "사각형":
        sw = st.number_input(
            "세면부 폭",
            0,
            5000,
            1300,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음"),
            key="sw_main",
        )
        sl = st.number_input(
            "세면부 길이",
            0,
            6000,
            1500,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음"),
            key="sl_main",
        )
        shw = st.number_input(
            "샤워부 폭",
            0,
            5000,
            800,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음"),
            key="shw_main",
        )
        shl = st.number_input(
            "샤워부 길이",
            0,
            6000,
            900,
            step=10,
            disabled=(central == "Yes" or btype == "구분없음"),
            key="shl_main",
        )
    else:
        st.caption("코너형일 경우 벽폭 분해치/영역 치수는 벽·천장 탭에서 세부 입력")

    st.divider()
    mgmt_rate_pct = st.number_input(
        "관리비율(%)", 0.0, 100.0, 25.0, step=0.5, key="mgmt_rate"
    )
    pve_kind = st.radio(
        "PVE 유형", ["일반형(+380mm)", "주거약자(+480mm)"], key="pve_kind_radio"
    )

    st.divider()
    # ===== 바닥판 규격표 업로더 (auto-apply) =====
    with st.expander("바닥판 규격 엑셀 (시트 권장명: '바닥판')", expanded=False):
        up_floor = st.file_uploader(
            "업로드 (.xlsx)",
            type=["xlsx"],
            key=f"floor_upload_{st.session_state['floor_upload_counter']}",
        )
        if up_floor is not None:
            b = up_floor.getvalue()
            if (st.session_state["floor_catalog_bytes"] is None) or (
                up_floor.name != st.session_state["floor_catalog_name"]
            ):
                st.session_state["floor_catalog_bytes"] = b
                st.session_state["floor_catalog_name"] = up_floor.name
                try:
                    st.toast(f"바닥판 규격 적용: {up_floor.name}")
                except Exception:
                    st.success(f"바닥판 규격 적용: {up_floor.name}")

        if st.session_state["floor_catalog_bytes"]:
            st.caption(f"현재 적용: **{st.session_state['floor_catalog_name']}**")
        else:
            st.caption("현재 적용: (없음)")

        if st.button("초기화", key="btn_reset_floor_catalog"):
            st.session_state["floor_catalog_bytes"] = None
            st.session_state["floor_catalog_name"] = None
            st.session_state["floor_upload_counter"] += 1
            st.rerun()

    # ===== 천장 카탈로그 업로더 (auto-apply) =====
    with st.expander("천장 카탈로그 엑셀 (시트: '천창판' / '천장판')", expanded=False):
        up_ceil = st.file_uploader(
            "업로드 (.xlsx)",
            type=["xlsx"],
            key=f"ceil_upload_{st.session_state['ceil_upload_counter']}",
        )
        if up_ceil is not None:
            b = up_ceil.getvalue()
            if (st.session_state["ceil_catalog_bytes"] is None) or (
                up_ceil.name != st.session_state["ceil_catalog_name"]
            ):
                st.session_state["ceil_catalog_bytes"] = b
                st.session_state["ceil_catalog_name"] = up_ceil.name
                try:
                    st.toast(f"카탈로그 적용: {up_ceil.name}")
                except Exception:
                    st.success(f"카탈로그 적용: {up_ceil.name}")

        if st.session_state["ceil_catalog_bytes"]:
            st.caption(f"현재 적용: **{st.session_state['ceil_catalog_name']}**")
        else:
            st.caption("현재 적용: (없음)")

        if st.button("초기화", key="btn_reset_ceil_catalog"):
            st.session_state["ceil_catalog_bytes"] = None
            st.session_state["ceil_catalog_name"] = None
            st.session_state["ceil_upload_counter"] += 1
            st.rerun()

# ================== 탭 ==================
tab_floor, tab_wall, tab_ceil, tab_sum = st.tabs(["바닥", "벽", "천장", "Summary"])

# ------------------ 1) 바닥 ------------------
with tab_floor:
    st.subheader("바닥 계산")
    st.write("규격표 기반 매칭(선택) + PVE 견적 산출")
    df_floor = None

    # 바닥 규격표가 적용되어 있으면 미리보기
    if st.session_state["floor_catalog_bytes"]:
        try:
            bio_floor = io.BytesIO(st.session_state["floor_catalog_bytes"])
            try:
                df_floor = pd.read_excel(
                    bio_floor, sheet_name="바닥판", engine="openpyxl"
                )
            except Exception:
                bio_floor.seek(0)
                df_floor = pd.read_excel(bio_floor, engine="openpyxl")  # 첫 시트
            st.dataframe(df_floor.head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"바닥 규격표 읽기 실패 → 매칭 없이 진행 ({e})")

    # (선택) 매칭 호출 — 실제 인자/반환은 네 모듈 시그니처에 맞춰 적용
    if df_floor is not None:
        if central == "Yes":
            st.caption("중앙배수 Yes → GRP(중앙배수) 우선 매칭")
            # 예: match_center_drain(df_floor, shape, btype, bw, bl)
        else:
            if shape == "사각형":
                # 예: match_non_center_rectangle(df_floor, btype, bw, bl, sw, sl, shw, shl)
                pass
            else:
                # 예: match_corner_shower(df_floor, bw, bl, sw, sl, shw, shl)
                pass

    # PVE 견적
    floor_is_pve = st.toggle(
        "바닥 소재: PVE로 가정", value=True, key="floor_is_pve_toggle"
    )
    if floor_is_pve:
        kind = "일반형" if "일반" in pve_kind else "주거약자"
        pve = pve_quote(bw, bl, mgmt_rate_pct / 100.0, kind=kind)
        st.write("PVE 견적:", pve)
        st.session_state["floor_meta"] = {
            "type": "PVE",
            "pve_detail": pve,
            "kind": kind,
        }
    else:
        st.session_state["floor_meta"] = {"type": "FRP/GRP"}

    # 개략 도식
    try:
        img = draw_bathroom(
            shape,
            bw,
            bl,
            None if (central == "Yes" or btype == "구분없음") else sw,
            None if (central == "Yes" or btype == "구분없음") else sl,
            None if (central == "Yes" or btype == "구분없음") else shw,
            None if (central == "Yes" or btype == "구분없음") else shl,
            central=central,
            btype=btype,
        )
        st.image(img, caption="개략 도식")
    except Exception:
        pass

# ------------------ 2) 벽 ------------------
with tab_wall:
    st.subheader("벽판 분할(타일 규격 기반)")
    tile = st.selectbox("타일 규격", ["300×600", "250×400"], key="tile_select")
    TH, TW = (300, 600) if tile == "300×600" else (250, 400)

    base_H = st.number_input("벽 높이(H, mm)", 1500, 4800, 2400, step=50, key="base_H")
    floor_type = st.session_state["floor_meta"].get("type", "FRP/GRP")
    H_eff = effective_height(base_H, floor_type)  # PVE면 +50
    st.write(f"유효 높이(H'): {H_eff} mm  (바닥:{floor_type})")

    W_face = st.number_input("대상 벽 폭(W, mm)", 600, 4800, bl, step=50, key="W_face")

    if st.button("벽 분할 계산", key="btn_wall_calc"):
        try:
            panels, rule_label = compute_layout(W_face, H_eff, TH, TW)
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
            st.info(f"규칙: {rule_label}")
            st.session_state["wall_result"] = {
                "panels": panels,
                "rule": rule_label,
                "tile": tile,
            }
        except Exception as e:
            st.error(f"분할 실패: {e}")

# ------------------ 3) 천장 ------------------
with tab_ceil:
    st.subheader("천장 최적화(카탈로그 기반)")

    # 사이드바에서 auto-apply된 바이트 사용
    if st.session_state["ceil_catalog_bytes"]:
        try:
            bio = io.BytesIO(st.session_state["ceil_catalog_bytes"])
            df_check, df_body, df_side = parse_catalog(bio)
        except Exception:
            st.warning("시트 자동 인식 실패 → 샘플 카탈로그 사용")
            df_check, df_body, df_side = sample_catalog()
    else:
        df_check, df_body, df_side = sample_catalog()

    st.write("점검구/바디/사이드 프리뷰")
    st.dataframe(df_check.head(8), use_container_width=True)
    st.dataframe(df_body.head(8), use_container_width=True)
    st.dataframe(df_side.head(8), use_container_width=True)

    st.divider()
    st.markdown("**치수 입력(직사각 기준)**")
    Wc = st.number_input("천장 폭 W (mm)", 600, 1900, 1300, step=10, key="ceil_w")
    Lc = st.number_input("천장 길이 L (mm)", 800, 4000, 1750, step=10, key="ceil_l")
    cut_cost = st.number_input(
        "컷 비용(원/컷)", 0, 100000, 3000, step=500, key="cut_cost"
    )

    if st.button("천장 최적화(직사각)", key="btn_ceil_rect"):
        result = optimize_rect(
            Wc, Lc, df_check, df_body, df_side, cut_cost, mgmt_rate_pct
        )
        st.write(result)
        st.session_state["ceil_result"] = result

    with st.expander("코너형(세면부/샤워부 분리)"):
        S_W = st.number_input("세면부 폭", 600, 1900, 1300, step=10, key="s_w_exp")
        S_L = st.number_input("세면부 길이", 800, 4000, 1750, step=10, key="s_l_exp")
        H_W = st.number_input("샤워부 폭", 400, 1900, 900, step=10, key="h_w_exp")
        H_L = st.number_input("샤워부 길이", 400, 4000, 900, step=10, key="h_l_exp")

        if st.button("천장 최적화(코너형)", key="btn_ceil_corner"):
            result2 = optimize_corner(
                S_W, S_L, H_W, H_L, df_check, df_body, df_side, cut_cost, mgmt_rate_pct
            )
            st.write(result2)
            st.session_state["ceil_result"] = result2

# ------------------ 4) Summary ------------------
with tab_sum:
    st.subheader("요약 / 내보내기")
    st.markdown("### 적용 소스")
    st.json(
        {
            "바닥 규격표": st.session_state.get("floor_catalog_name"),
            "천장 카탈로그": st.session_state.get("ceil_catalog_name"),
        }
    )

    st.markdown("### 바닥")
    st.json(st.session_state["floor_meta"])

    st.markdown("### 벽")
    wall_result = st.session_state["wall_result"]
    st.json(
        {
            "tile": wall_result.get("tile"),
            "rule": wall_result.get("rule"),
            "panels_count": len(wall_result.get("panels", [])),
        }
    )

    st.markdown("### 천장")
    st.json(
        st.session_state["ceil_result"]
        if st.session_state["ceil_result"]
        else {"status": "no-run"}
    )

    st.info("필요 시 CSV/엑셀로 내보내기 버튼 추가 가능")
