# pages/01_Floor_UI.py (수정됨: calculate_floor_panel 함수 활용)
import io
import streamlit as st
import pandas as pd
import ui_theme as ui
from floor_panel_final import (
    pve_quote,
    draw_bathroom,
    normalize_df,
    calculate_floor_panel,
)  # 👈 calculate_floor_panel, normalize_df 추가 임포트

st.set_page_config(page_title="UBR · 바닥", layout="wide")
ui.apply()  # 👈 스타일 적용
ui.hero("바닥 계산", "규격표/치수 기반 PVE 견적 · 개략 도식")

# 세션 키
for k, v in {"F_bytes": None, "F_name": None, "F_counter": 0}.items():
    st.session_state.setdefault(k, v)

left, right = st.columns([1, 1], gap="large")

# ---------------- 입력 카드 ----------------
with left:
    with ui.card("입력", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            shape = st.radio(
                "형태", ["사각형", "코너형"], horizontal=True, key="F_shape"
            )
            central = st.radio(
                "중앙배수", ["No", "Yes"], horizontal=True, key="F_central"
            )
            btype = st.radio("유형", ["샤워형", "욕조형", "구분없음"], key="F_type")
        with c2:
            units = st.number_input("공사 세대수", 1, 100000, 100, key="F_units")
            mgmt = st.number_input(
                "관리비율(%)", 0.0, 100.0, 25.0, step=0.5, key="F_mgmt"
            )
            mgmt_rate = mgmt / 100.0  # 관리 비율 (0.0 ~ 1.0)

        # 세면/샤워부 활성화 조건
        disable_sink_shower = (
            (central == "Yes")
            or (btype == "구분없음")
            or (shape == "코너형" and btype != "샤워형")
        )

        col1, col2 = st.columns(2)
        with col1:
            bw = st.number_input(
                "욕실 폭 BW (mm)", 400, 6000, 1500, step=10, key="F_bw"
            )
            sw = st.number_input(
                "세면부 폭 (mm)",
                0,
                6000,
                1300 if not disable_sink_shower else 0,  # 비활성 시 기본값 0
                step=10,
                disabled=disable_sink_shower,
                key="F_sw",
            )
            shw = st.number_input(
                "샤워부 폭 (mm)",
                0,
                6000,
                800 if not disable_sink_shower else 0,  # 비활성 시 기본값 0
                step=10,
                disabled=disable_sink_shower,
                key="F_shw",
            )
        with col2:
            bl = st.number_input(
                "욕실 길이 BL (mm)", 400, 6000, 2200, step=10, key="F_bl"
            )
            sl = st.number_input(
                "세면부 길이 (mm)",
                0,
                6000,
                1500 if not disable_sink_shower else 0,  # 비활성 시 기본값 0
                step=10,
                disabled=disable_sink_shower,
                key="F_sl",
            )
            shl = st.number_input(
                "샤워부 길이 (mm)",
                0,
                6000,
                900 if not disable_sink_shower else 0,  # 비활성 시 기본값 0
                step=10,
                disabled=disable_sink_shower,
                key="F_shl",
            )

        # 비활성 시 None으로 변환 (calculate_floor_panel로 전달하기 위해)
        sw_calc = sw if not disable_sink_shower else None
        sl_calc = sl if not disable_sink_shower else None
        shw_calc = shw if not disable_sink_shower else None
        shl_calc = shl if not disable_sink_shower else None

        ui.divider()
        st.markdown(
            "**바닥판 규격 엑셀 (선택)** <span class='muted'>(시트명 '바닥판' 권장)</span>",
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

        run = st.button("바닥 계산", type="primary", use_container_width=True)

# ---------------- 결과 카드 ----------------
with right:
    with ui.card("결과", expanded=True):
        if run:
            pve_kind_select = st.selectbox(  # PVE 유형을 결과 실행 후에 선택하도록 이동
                "PVE 유형", ["일반형(+380mm)", "주거약자(+480mm)"], key="F_pve_kind"
            )
            pve_kind = "일반형" if pve_kind_select.startswith("일반") else "주거약자"

            # 엑셀 데이터 로딩
            df_norm = pd.DataFrame()
            if st.session_state["F_bytes"]:
                try:
                    bio = io.BytesIO(st.session_state["F_bytes"])
                    # 엑셀을 읽고 정규화
                    raw_df = pd.read_excel(bio, sheet_name="바닥판", engine="openpyxl")
                    df_norm = normalize_df(raw_df)
                    st.dataframe(df_norm.head(12), use_container_width=True, height=200)
                except Exception as e:
                    st.warning(f"바닥 규격표 읽기 또는 정규화 실패: {e}")
                    df_norm = pd.DataFrame()  # 실패 시 빈 DataFrame 유지

            # 계산 실행 (규격표 데이터가 없으면 calculate_floor_panel 내부에서 PVE 강제 선택됨)
            if df_norm.empty and st.session_state["F_bytes"]:
                # 엑셀 파일은 올렸으나, 시트명 문제 등으로 데이터 로딩 실패 시
                st.error(
                    "엑셀 파일 로딩/정규화에 실패했습니다. PVE 견적으로만 진행합니다."
                )

            result = calculate_floor_panel(
                df=df_norm,
                units=units,
                central=central,
                shape=shape,
                btype=btype,
                bw=bw,
                bl=bl,
                sw=sw_calc,
                sl=sl_calc,
                shw=shw_calc,
                shl=shl_calc,
                mgmt_rate=mgmt_rate,
                pve_kind=pve_kind,
            )

            # 도식
            try:
                # draw_bathroom에 sw, sl, shw, shl은 mm값 그대로 전달
                img = draw_bathroom(
                    shape, bw, bl, sw_calc, sl_calc, shw_calc, shl_calc, central, btype
                )
                st.image(img, caption="개략 도식", width=480)
            except Exception as e:
                st.error(f"도형 렌더링 오류: {e}")

            # 결과 출력
            base_subtotal = result["base_subtotal"]
            mgmt_total = result["mgmt_total"]
            mgmt_rate_pct = mgmt
            result_kind = result["result_kind"]
            decision_log = result["decision_log"]

            st.subheader("선택된 바닥판")
            st.markdown(f"**재질**: **{result_kind}**")
            st.markdown(f"**소계(원)**: **{base_subtotal:,}**")
            st.markdown(
                f"**관리비 포함 소계(원)**: **{mgmt_total:,}** (관리비율 {mgmt_rate_pct:.1f}%)"
            )

            st.info("결정 과정", icon="ℹ️")
            st.write("\n".join([f"- {x}" for x in decision_log]))

            st.success("계산 완료 ✅")

        else:
            st.info("좌측에서 입력/업로드 후 ‘바닥 계산’을 누르세요.")
