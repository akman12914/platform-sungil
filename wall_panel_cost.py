# -*- coding: utf-8 -*-
# wall_panel_cost_avg_only.py
# Streamlit: 타일 벽판넬 생산원가계(AD) "평균 1장" 기준 계산
# - 입력된 (폭, 높이)×수량 패널 리스트로 총면적/총둘레를 구한 뒤, 평균치로 원가 1개만 산출
# 실행: streamlit run wall_panel_cost_avg_only.py

import io
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st


@dataclass
class ExcelConsts:
    frame_unit_price: float        # 원/m
    pu_unit_price: float           # 원/㎡
    clip_unit_price: float         # 원/판넬(1세트)
    equip_depr_unit: float         # 원/판넬
    manuf_overhead_unit: float     # 원/판넬
    tile_mgmt_unit_price: float    # 원/타일(장)
    ship_rack_unit: float          # 원/판넬
    labor_cost_per_day: float      # 원/일
    loss_rate: float = 1.02
    wall_height_default_m: float = 2.3
    prod_qty_le_1_5: int = 325
    prod_qty_1_51_1_89: int = 300
    prod_qty_ge_1_9: int = 275


def _to_num(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return float("nan")
    return float(s.replace(",", ""))


def _normalize_two_col_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    if {"variable", "value"}.issubset(set(cols)):
        return df
    if len(cols) < 2:
        raise ValueError("상수 시트('벽판')는 최소 2개 컬럼(변수명/값)이 필요합니다.")
    df2 = df.iloc[:, :2].copy()
    df2.columns = ["variable", "value"]
    return df2


def load_consts_from_sheet(excel_bytes: bytes, angle: int, sheet_name: str = "벽판") -> ExcelConsts:
    raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)
    df = _normalize_two_col_df(raw)

    kv: Dict[str, float] = {}
    for _, r in df.iterrows():
        k = str(r["variable"]).strip()
        if not k or k.lower() in {"nan", "none"}:
            continue
        kv[k] = _to_num(r["value"])

    def req(key: str) -> float:
        if key not in kv or pd.isna(kv[key]):
            raise ValueError(f"상수 시트('{sheet_name}')에 필수 변수 '{key}'가 없습니다(또는 값이 비어있음).")
        return float(kv[key])

    def opt(key: str, default: float) -> float:
        v = kv.get(key, default)
        return default if pd.isna(v) else float(v)

    return ExcelConsts(
        frame_unit_price=req(f"프레임단가_{angle}각"),
        pu_unit_price=req(f"P_U단가_{angle}각"),
        clip_unit_price=req("조립클립단가"),
        equip_depr_unit=req("설비감가비"),
        manuf_overhead_unit=req("제조경비_판넬당"),
        tile_mgmt_unit_price=req("타일관리비_단가"),
        ship_rack_unit=req("출고_렉입고_단가"),
        labor_cost_per_day=req("생산인건비_일단가"),
        loss_rate=opt("프레임_LOSS_배수", 1.02),
        wall_height_default_m=opt("벽체높이_기본_m", 2.3),
        prod_qty_le_1_5=int(opt("기준생산량_1_5이하", 325)),
        prod_qty_1_51_1_89=int(opt("기준생산량_1_51_1_89", 300)),
        prod_qty_ge_1_9=int(opt("기준생산량_1_9이상", 275)),
    )


def gen_random_panels(
    n_rows: int,
    wall_h_mm: float,
    w_min: float = 300,
    w_max: float = 1200,
    qty_min: int = 1,
    qty_max: int = 3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
    rows = []
    for _ in range(n_rows):
        w = round(random.uniform(w_min, w_max), 0)
        h = round(wall_h_mm, 0)
        qty = random.randint(qty_min, qty_max)
        rows.append({"패널폭(mm)": int(w), "패널높이(mm)": int(h), "수량": qty})
    return pd.DataFrame(rows)


def production_qty_from_avg_area(avg_area_m2: float, consts: ExcelConsts) -> int:
    if avg_area_m2 <= 1.5:
        return consts.prod_qty_le_1_5
    if avg_area_m2 <= 1.89:
        return consts.prod_qty_1_51_1_89
    return consts.prod_qty_ge_1_9


def compute_avg_cost(
    panels: pd.DataFrame,
    consts: ExcelConsts,
    bath_type: str,
    zendae_step: bool,
    zendae_h_mm: float,
    wall_h_mm: float,
    tiles_per_panel: float,
):
    df = panels.copy()
    df["패널폭(mm)"] = pd.to_numeric(df["패널폭(mm)"], errors="coerce")
    df["패널높이(mm)"] = pd.to_numeric(df["패널높이(mm)"], errors="coerce")
    df["수량"] = pd.to_numeric(df["수량"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["패널폭(mm)", "패널높이(mm)"])
    df = df[df["수량"] > 0]
    if df.empty:
        raise ValueError("패널 리스트가 비어있습니다(수량>0).")

    total_panels = int(df["수량"].sum())

    # mm를 m로 변환하여 면적 계산
    df["패널폭(m)"] = df["패널폭(mm)"] / 1000.0
    df["패널높이(m)"] = df["패널높이(mm)"] / 1000.0
    zendae_h_m = zendae_h_mm / 1000.0
    wall_h_m = wall_h_mm / 1000.0

    df["패널면적(㎡)"] = df["패널폭(m)"] * df["패널높이(m)"]
    total_area = float((df["패널면적(㎡)"] * df["수량"]).sum())
    avg_area = total_area / total_panels

    df["패널둘레(m/장)"] = 2.0 * (df["패널폭(m)"] + df["패널높이(m)"])
    base_frame_total = float((df["패널둘레(m/장)"] * df["수량"]).sum())

    if bath_type == "사각형" and (not zendae_step):
        add_len = 0.0
    elif bath_type == "사각형" and zendae_step:
        add_len = 2.0 * zendae_h_m
    elif bath_type == "코너형" and (not zendae_step):
        add_len = wall_h_m
    else:
        add_len = zendae_h_m + wall_h_m

    frame_total = base_frame_total + add_len
    frame_total_loss = frame_total * consts.loss_rate
    avg_frame_len_loss = frame_total_loss / total_panels

    frame_cost = avg_frame_len_loss * consts.frame_unit_price
    pu_cost = avg_area * consts.pu_unit_price
    clip_cost = consts.clip_unit_price
    material_M = frame_cost + pu_cost + clip_cost

    prod_qty = production_qty_from_avg_area(avg_area, consts)
    sets_per_panel = prod_qty / total_panels
    labor_P = consts.labor_cost_per_day / sets_per_panel

    equip_S = consts.equip_depr_unit
    manuf_V = consts.manuf_overhead_unit
    tile_Y = tiles_per_panel * consts.tile_mgmt_unit_price
    ship_AB = consts.ship_rack_unit

    cost_AD = material_M + labor_P + equip_S + manuf_V + tile_Y + ship_AB
    total_set_cost = cost_AD * total_panels

    summary = {
        "총판넬수": float(total_panels),
        "총면적(㎡)": float(total_area),
        "평균면적(㎡/장)": float(avg_area),
        "기본프레임총길이(m)": float(base_frame_total),
        "추가프레임길이(m)": float(add_len),
        "Loss적용프레임총길이(m)": float(frame_total_loss),
        "후레임평균(m/장,Loss)": float(avg_frame_len_loss),
        "생산량(기준)": float(prod_qty),
        "판넬1장당_평균가공세트수": float(sets_per_panel),
        "판넬1장당_생산인건비(P)": float(labor_P),
        "판넬1장당_생산원가계(AD)": float(cost_AD),
        "욕실1세트_생산원가계(AD)": float(total_set_cost),
    }

    breakdown = pd.DataFrame(
        [
            ("재료비(M)-프레임", "원/장", frame_cost, frame_cost * total_panels),
            ("재료비(M)-P/U", "원/장", pu_cost, pu_cost * total_panels),
            ("재료비(M)-조립클립", "원/장", clip_cost, clip_cost * total_panels),
            ("재료비(M) 합계", "원/장", material_M, material_M * total_panels),
            ("생산인건비(P)", "원/장", labor_P, labor_P * total_panels),
            ("설비감가비(S)", "원/장", equip_S, equip_S * total_panels),
            ("제조경비(V)", "원/장", manuf_V, manuf_V * total_panels),
            ("타일관리비(Y)", "원/장", tile_Y, tile_Y * total_panels),
            ("출고·렉입고비(AB)", "원/장", ship_AB, ship_AB * total_panels),
            ("생산원가계(AD)", "원/장", cost_AD, total_set_cost),
        ],
        columns=["항목", "단위", "판넬 1장(평균) 원가", "욕실 1세트(총) 원가"],
    )

    return df, summary, breakdown


st.set_page_config(page_title="타일 벽판넬 생산원가계(평균 1장) 계산", layout="wide")
st.title("타일 벽판넬 생산원가계(AD) 계산기 — 평균 1장 기준(원가 1개만 산출)")

with st.sidebar:
    st.header("① 상수 엑셀 업로드")
    const_uploaded = st.file_uploader("상수 엑셀 업로드(.xlsx) — '벽판' 시트(variable/value)", type=["xlsx"])

    st.header("② 입력 조건")
    angle = st.radio("프레임 선정", [15, 16, 19], format_func=lambda x: f"{x}각", horizontal=True)

    if const_uploaded is None:
        st.info("상수 엑셀을 업로드하세요. (시트명: '벽판')")
        st.stop()

    try:
        consts = load_consts_from_sheet(const_uploaded.read(), angle=int(angle), sheet_name="벽판")
    except Exception as e:
        st.error(f"상수 로딩 실패: {e}")
        st.stop()

    bath_type = st.radio("욕실형태유형", ["사각형", "코너형"], horizontal=True)
    zendae_step = st.radio("젠다이 단차여부", ["없음", "있음"], horizontal=True) == "있음"

    wall_h_mm = st.number_input(
        "벽 높이 H (mm)",
        min_value=1800,
        max_value=3000,
        value=int(consts.wall_height_default_m * 1000),
        step=10,
    )

    zendae_h_mm = 0.0
    if zendae_step:
        zendae_h_mm = st.number_input("젠다이 높이 (mm)", min_value=50, max_value=1000, value=200, step=10)

    tiles_per_panel = st.number_input(
        "벽판 1장당 타일의 개수 (소수점 1자리)",
        min_value=0.0,
        value=10.5,
        step=0.1,
        format="%.1f",
    )

    st.divider()
    st.header("③ 패널 리스트 생성/편집")
    n_rows = st.slider("패널 행 개수(임의 생성)", min_value=10, max_value=30, value=15, step=1)
    seed = st.number_input("랜덤 시드(재현용)", min_value=0, max_value=999999, value=1234, step=1)
    gen_btn = st.button("패널 임의 생성", use_container_width=True)

    st.caption("고급 설정")
    loss_rate = st.number_input("프레임 Loss율(배수)", min_value=1.0, max_value=1.2, value=float(consts.loss_rate), step=0.005)
    consts.loss_rate = float(loss_rate)

if "panels_df" not in st.session_state:
    st.session_state.panels_df = gen_random_panels(n_rows=15, wall_h_mm=float(consts.wall_height_default_m * 1000), seed=1234)

if gen_btn:
    st.session_state.panels_df = gen_random_panels(n_rows=int(n_rows), wall_h_mm=float(wall_h_mm), seed=int(seed))

st.subheader("패널 리스트(편집 가능)")
edited = st.data_editor(
    st.session_state.panels_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "패널폭(mm)": st.column_config.NumberColumn(min_value=100, max_value=2000, step=10, format="%d"),
        "패널높이(mm)": st.column_config.NumberColumn(min_value=100, max_value=3500, step=10, format="%d"),
        "수량": st.column_config.NumberColumn(min_value=1, max_value=999, step=1),
    },
)
st.session_state.panels_df = edited

try:
    clean_df, summary, breakdown = compute_avg_cost(
        panels=st.session_state.panels_df,
        consts=consts,
        bath_type=bath_type,
        zendae_step=zendae_step,
        zendae_h_mm=float(zendae_h_mm),
        wall_h_mm=float(wall_h_mm),
        tiles_per_panel=float(tiles_per_panel),
    )
except Exception as e:
    st.error(f"계산 실패: {e}")
    st.stop()

st.divider()
st.subheader("평균치 기반 결과(원가 1개만 산출)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 판넬 수", f"{summary['총판넬수']:,.0f}")
c1.metric("총 면적(㎡)", f"{summary['총면적(㎡)']:,.3f}")
c2.metric("평균 면적(㎡/장)", f"{summary['평균면적(㎡/장)']:,.3f}")
c2.metric("후레임 평균(m/장,Loss)", f"{summary['후레임평균(m/장,Loss)']:,.3f}")
c3.metric("생산량(기준)", f"{summary['생산량(기준)']:,.0f}")
c3.metric("판넬 1장당 생산인건비(P)", f"{summary['판넬1장당_생산인건비(P)']:,.0f}")
c4.metric("판넬 1장당 생산원가계(AD)", f"{summary['판넬1장당_생산원가계(AD)']:,.0f}")
c4.metric("욕실 1세트 생산원가계(AD)", f"{summary['욕실1세트_생산원가계(AD)']:,.0f}")

with st.expander("프레임 길이 상세", expanded=False):
    st.write(f"- 기본프레임 총길이(m): **{summary['기본프레임총길이(m)']:,.3f}**")
    st.write(f"- 추가프레임 길이(m): **{summary['추가프레임길이(m)']:,.3f}**")
    st.write(f"- Loss 적용 프레임 총길이(m): **{summary['Loss적용프레임총길이(m)']:,.3f}**")
    st.caption("※ 평균 프레임 길이(장당)는 (Loss적용프레임총길이 / 총판넬수) 입니다.")

st.subheader("원가 구성(평균 1장 vs 욕실 1세트)")
st.dataframe(
    breakdown,
    use_container_width=True,
    height=420,
    column_config={
        "판넬 1장(평균) 원가": st.column_config.NumberColumn(format="%.0f"),
        "욕실 1세트(총) 원가": st.column_config.NumberColumn(format="%.0f"),
    },
)

st.subheader("입력 패널(정리된 리스트)")
st.dataframe(clean_df, use_container_width=True, height=320)
