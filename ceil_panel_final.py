# app_ceiling_full.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools
import re, unicodedata, difflib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import os, json
from datetime import datetime
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def _save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def candidate_to_dict(c: Candidate) -> Dict[str, Any]:
    return {
        "pattern": list(c.pattern),
        "oriented": [
            {
                "kind": o.panel.kind,
                "name": o.panel.name,
                "cw": o.cw,
                "cl": o.cl,
                "rotated": o.rotated,
                "panel": {
                    "name": o.panel.name,
                    "kind": o.panel.kind,
                    "width": o.panel.width,
                    "length": o.panel.length,
                    "price": o.panel.price,
                },
            }
            for o in c.oriented
        ],
        "width_cuts": list(c.width_cuts),
        "length_cut_last": int(c.length_cut_last),
        "material_cost": int(c.material_cost),
        "cut_cost": int(c.cut_cost),
        "total_cost": int(c.total_cost),
    }

# --- design refresh (prettier inline) ---
import streamlit as st


def _design_refresh():

    st.markdown(
        """
    <style>
      :root{
        /* Sidebar dark palette */
        --sb-bg:#0b1220;         /* 다크 네이비 */
        --sb-fg:#e2e8f0;         /* 본문 텍스트 */
        --sb-muted:#475569;      /* 🔸보조 텍스트: 더 밝게/진하게 */
        --sb-line:#1f2a44;

        /* Accents: 시안 → 더 어울리는 Teal 계열 */
        --accent:#f1f5f9;   /* 거의 흰색 (상단) */
        --accent-2:#cbd5e1; /* 밝은 회색 (하단) */

        /* Main content neutrals */
        --ink:#0f172a;
        --muted:#475569;
        --line:#e2e8f0;
      }

      /* Sidebar Dark */
      section[data-testid="stSidebar"]{
        background:var(--sb-bg)!important; color:var(--sb-fg)!important;
        border-right:1px solid var(--sb-line);
      }
      section[data-testid="stSidebar"] *{ color:var(--sb-fg)!important; }
      section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{
        color:var(--sb-fg)!important;
      }

      /* 🔸보조 텍스트/라벨: 더 선명 + 약간 굵게 */
      section[data-testid="stSidebar"] .stMarkdown p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] .stSelectbox label{
        color:var(--muted)!important;
        font-weight:600!important;
      }

      /* Inputs in sidebar */
      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] textarea,
      section[data-testid="stSidebar"] select,
      section[data-testid="stSidebar"] .stTextInput input,
      section[data-testid="stSidebar"] .stNumberInput input{
        background:rgba(255,255,255,0.06)!important;
        border:1px solid var(--sb-line)!important;
        color:var(--muted)!important;
      }

      /* 🔧 Slider cutoff fix */
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ padding-right:12px; }
      section[data-testid="stSidebar"] div[data-testid="stSlider"]{
        padding-right:12px; margin-right:2px; overflow:visible;
      }
      section[data-testid="stSidebar"] div[role="slider"]{
        box-shadow:0 0 0 2px rgba(20,184,166,0.25); border-radius:999px;
      }

      /* ✅ Radio: 색/정렬 깔끔하게 (red → teal, 정중앙 정렬) */
      /* Streamlit 라디오 인풋 컬러를 액센트로 통일 */
      input[type="radio"]{ accent-color: var(--accent); }
      /* 라벨/원형이 수직 중앙 정렬되도록 라벨 플렉스 정렬 */
      div[role="radiogroup"] label{
        display:flex; align-items:center; gap:.5rem;
        line-height:1.2; margin: .1rem 0;
      }
      /* 일부 환경에서 라디오 원이 1px 내려가 보이는 현상 보정 */
      div[role="radiogroup"] input[type="radio"]{
        transform: translateY(0px);
      }

      /* Buttons (sidebar/main 공통) */
      section[data-testid="stSidebar"] .stButton>button,
      [data-testid="stAppViewContainer"] .stButton>button{
        background:linear-gradient(180deg,var(--accent),var(--accent-2))!important;
        color:#001018!important;
        border:0!important; font-weight:800!important; letter-spacing:.2px;
        border-radius:10px; padding:.55rem 1rem;
      }
      section[data-testid="stSidebar"] .stButton>button:hover,
      [data-testid="stAppViewContainer"] .stButton>button:hover{
        filter:brightness(1.05);
      }

      /* 이미지 여백 (겹침 방지) */
      [data-testid="stImage"]{ margin:6px 0 18px!important; }
      [data-testid="stImage"] img{ display:block; }

        span[label="app main"] {
      font-size: 0 !important;          /* 기존 글자 숨김 */
      position: relative;
  }
  span[label="app main"]::after {
      content: "메인";                  /* 원하는 표시 이름 */
      font-size: 1rem !important;       /* 기본 폰트 크기로 복원 */
      color: #fff !important;           /* 사이드바 글씨 색 (흰색) */
      font-weight: 700 !important;      /* 굵게 */
      position: absolute;
      left: 0;
      top: 0;
  }

        /* 🔹 FileUploader 전체 영역 */
      section[data-testid="stFileUploaderDropzone"] {
          border: 2px dashed var(--sb-line) !important;
          background: rgba(255,255,255,0.03) !important;
          color: var(--sb-muted) !important;
          border-radius: 10px !important;
          padding: 12px !important;
      }

      /* 아이콘 색상 */
      section[data-testid="stFileUploaderDropzone"] svg {
          color: var(--sb-muted) !important;
          fill: var(--sb-muted) !important;
      }

      /* 안내 텍스트 */
      section[data-testid="stFileUploaderDropzone"] span {
          color: var(--sb-muted) !important;
          font-weight: 600 !important;
      }

      /* 버튼 */
      section[data-testid="stFileUploaderDropzone"] button {
          background: linear-gradient(180deg,var(--accent),var(--accent-2)) !important;
          color: #001018 !important;
          border: 0 !important;
          font-weight: 700 !important;
          border-radius: 8px !important;
          padding: .4rem .9rem !important;
      }
      section[data-testid="stFileUploaderDropzone"] button:hover {
          filter: brightness(1.05);
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


# --- end design refresh ---

_design_refresh()


import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 설정 / 상수
# =========================================================
CUT_COST_DEFAULT = 3000
MGMT_RATIO_DEFAULT = 25.0
DOUBLE_CHECK_NAMES = {"SI-7", "SI-8", "SI-9"}  # 점검구 ×2 자동 적용
MAX_RECT_CANVAS_W = 540  # 화면 1/3 정도
MAX_RECT_CANVAS_H = 360


FLOOR_DONE_KEY = "floor_done"
FLOOR_RESULT_KEY = "floor_result"

WALL_DONE_KEY  = "wall_done"
WALL_RESULT_KEY = "wall_result"

CEIL_DONE_KEY  = "ceil_done"
CEIL_RESULT_KEY = "ceil_result"



# =========================================================
# 데이터 모델
# =========================================================
@dataclass(frozen=True)
class Panel:
    kind: str  # "B" or "S"
    name: str
    width: int  # mm
    length: int  # mm
    price: int  # 원


@dataclass
class Oriented:
    panel: Panel
    cw: int  # 배치폭 (회전 반영)
    cl: int  # 배치길이 (회전 반영)
    rotated: bool


@dataclass
class Candidate:
    pattern: List[str]  # ex) ["B"], ["B","S"], ["B","S","S"], ...
    oriented: List[Oriented]  # 패턴과 같은 길이
    width_cuts: List[int]  # 각 판 폭컷(0/1)
    length_cut_last: int  # 마지막 장 길이컷(0/1)
    material_cost: int
    cut_cost: int
    total_cost: int


# =========================================================
# 유틸 - 숫자/문자 파싱 & 정규화
# =========================================================
def _to_int(x, default=0):
    if pd.isna(x):
        return default
    s = str(x).replace(",", "").strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _norm_key(s: str) -> str:
    """제품명 매칭 키 정규화: NFKC, 소문자, 공백/NBSP 제거, 다양한 하이픈 통일"""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = t.replace("\u00a0", " ").strip()
    for h in ["‐", "-", "‒", "–", "—", "−", "﹘", "－"]:
        t = t.replace(h, "-")
    t = t.replace(" ", "")
    return t


# =========================================================
# 샘플 카탈로그 (엑셀 미업로드 시 사용)
# =========================================================
def sample_catalog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_check = pd.DataFrame(
        {
            "제품명": [
                "SI-2",
                "SI-3",
                "SI-4",
                "SI-5",
                "SI-6",
                "SI-7",
                "SI-8",
                "SI-9",
                "SI-10",
                "SI-11",
            ],
            "폭": [700, 700, 700, 500, 700, 450, 450, 450, 650, 750],
            "길이": [900, 900, 900, 650, 900, 450, 450, 450, 900, 900],
            "가격(원)": [8586, 8586, 8586, 6297, 8586, 4728, 4728, 4728, 8175, 8185],
        }
    )
    df_body = pd.DataFrame(
        {
            "제품명": [
                "SI-2",
                "SI-3",
                "SI-4",
                "SI-5",
                "SI-6",
                "SI-7",
                "SI-8",
                "SI-9",
                "SI-10",
                "SI-11",
            ],
            "폭": [1300, 1300, 1350, 1350, 1450, 1000, 1200, 1200, 1370, 1260],
            "길이": [2000, 1750, 1750, 1750, 1750, 1750, 1750, 2000, 1850, 1850],
            "가격(원)": [
                24877,
                19467,
                20465,
                22778,
                22091,
                22305,
                25854,
                31177,
                22091,
                21026,
            ],
        }
    )
    df_side = pd.DataFrame(
        {
            "제품명": [
                "700",
                "800",
                "900a",
                "900b",
                "1000",
                "1200",
                "2000",
                "750",
                "1100",
            ],
            "폭": [700, 800, 900, 900, 1000, 1200, 1200, 750, 1100],
            "길이": [1750, 1750, 1750, 960, 1750, 1750, 2000, 1850, 1850],
            "가격(원)": [14110, 15954, 18684, 10786, 19905, 23454, 28777, 14528, 20190],
        }
    )
    return df_check, df_body, df_side


# =========================================================
# 카탈로그 파싱 (엑셀 시트: '천창판' 또는 '천장판')
# =========================================================
def _norm_col(k: str) -> Optional[str]:
    k2 = re.sub(r"\s+", "", str(k)).lower()
    if k2 in ("제품명", "제품", "품명", "item", "product", "name"):
        return "제품명"
    if k2 in ("폭", "가로", "width", "w"):
        return "폭"
    if k2 in ("길이", "세로", "length", "l"):
        return "길이"
    if k2 in ("소계", "가격", "price", "금액", "단가", "합계", "총액"):
        return "가격(원)"
    return None


def _extract_section(df_raw: pd.DataFrame, title: str) -> pd.DataFrame:
    df = df_raw.copy()
    title_row = title_col = None
    for r in range(min(15, len(df))):
        for c in range(df.shape[1]):
            v = str(df.iat[r, c]).strip() if pd.notna(df.iat[r, c]) else ""
            if v == title:
                title_row, title_col = r, c
                break
        if title_row is not None:
            break
    if title_row is None:
        return pd.DataFrame(columns=["제품명", "폭", "길이", "가격(원)"])

    header_row = title_row + 1
    raw_cols = []
    for c in range(title_col, title_col + 4):
        v = (
            str(df.iat[header_row, c]).strip()
            if pd.notna(df.iat[header_row, c])
            else ""
        )
        raw_cols.append(v if v else f"col{c-title_col+1}")

    rename = {}
    for i, k in enumerate(raw_cols):
        rename[k] = _norm_col(k) or ["제품명", "폭", "길이", "가격(원)"][i]

    data = df.iloc[header_row + 1 :, title_col : title_col + 4].copy()
    data.columns = [rename[k] for k in raw_cols]

    if "가격(원)" not in data.columns and "소계" in data.columns:
        data.rename(columns={"소계": "가격(원)"}, inplace=True)
    if "제품명" not in data.columns:
        data["제품명"] = ""

    for col in ["폭", "길이", "가격(원)"]:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    if title == "점검구":
        data = data[~data["제품명"].isna()].copy()
    else:
        data = data.dropna(subset=["폭", "길이"])

    data["제품명"] = data["제품명"].astype(str).str.strip()
    return data.reset_index(drop=True)[["제품명", "폭", "길이", "가격(원)"]]


def parse_catalog(uploaded) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        df_raw = pd.read_excel(uploaded, sheet_name="천창판", header=None)
    except Exception:
        df_raw = pd.read_excel(uploaded, sheet_name="천장판", header=None)
    df_check = _extract_section(df_raw, "점검구")
    df_body = _extract_section(df_raw, "바디판넬")
    df_side = _extract_section(df_raw, "사이드판넬")
    return df_check, df_body, df_side


# =========================================================
# 패널 변환/정리
# =========================================================
def filter_valid(df: pd.DataFrame, b_req: int) -> pd.DataFrame:
    """길이 ≥ b_req, 그리고 길이 ≥ 폭(긴 변이 길이축)"""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns)
    d = df.copy()
    return d[(d["길이"] >= b_req) & (d["길이"] >= d["폭"])].reset_index(drop=True)


def df_to_panels(df: pd.DataFrame, kind: str) -> List[Panel]:
    out: List[Panel] = []
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        w, l, p = _to_int(r["폭"]), _to_int(r["길이"]), _to_int(r["가격(원)"])
        name = str(r["제품명"]).strip()
        if w > 0 and l > 0:
            out.append(Panel(kind, name, w, l, p))
    return out


def oriented_variants(p: Panel) -> List[Oriented]:
    """0°/90° 두 방향 모두 반환"""
    return [
        Oriented(panel=p, cw=p.width, cl=p.length, rotated=False),
        Oriented(panel=p, cw=p.length, cl=p.width, rotated=True),
    ]


# =========================================================
# 점검구 가격 조회(바디명과 동일 모델)
# =========================================================
def get_check_price(df_check: pd.DataFrame, body_name: str) -> Tuple[int, bool]:
    """(점검구 가격, is_double) 반환"""
    df = df_check.copy()
    df["__raw"] = df["제품명"].astype(str)
    df["__key"] = df["__raw"].apply(_norm_key)
    key = _norm_key(body_name)

    hit = df.loc[df["__key"] == key]
    if hit.empty:
        hit = df.loc[df["__key"].str.contains(key, na=False)]
    if hit.empty:
        # fuzzy
        choices = df["__key"].tolist()
        near = difflib.get_close_matches(key, choices, n=1, cutoff=0.8)
        if near:
            hit = df.loc[df["__key"] == near[0]]

    if hit.empty:
        return 0, False

    price = _to_int(hit.iloc[0]["가격(원)"])
    is_double = (
        body_name.split("(")[0].strip() in DOUBLE_CHECK_NAMES
        or body_name.strip() in DOUBLE_CHECK_NAMES
    )
    return price, is_double


# =========================================================
# 사각형 최적화 (B/S 카탈로그 전체 전수탐색)
# =========================================================
def eval_pattern_rect(
    Wc: int,
    Lc: int,
    pattern: List[str],
    bodies: List[Panel],
    sides: List[Panel],
    cut_cost: int,
) -> List[Candidate]:
    """패턴에 맞춰 모든 조합(회전 포함) 전수. 폭은 각 패널 cw ≥ Wc, 길이는 마지막만 컷 허용."""

    # 각 타입별 '폭 적합' 오리엔트 필터
    def valid_orients(kind: str) -> List[Oriented]:
        base = bodies if kind == "B" else sides
        out: List[Oriented] = []
        for p in base:
            for o in oriented_variants(p):
                if o.cw >= Wc:  # 폭 못 가로지르면 불가
                    out.append(o)
        return out

    pools = [valid_orients(k) for k in pattern]
    if any(len(pool) == 0 for pool in pools):
        return []

    candidates: List[Candidate] = []
    for combo in itertools.product(*pools):
        oriented_list: List[Oriented] = list(combo)

        # 길이 합/컷 규칙 체크
        acc = 0
        ok = True
        for i, o in enumerate(oriented_list):
            if i < len(oriented_list) - 1:
                if (
                    acc + o.cl >= Lc
                ):  # 중간 판이 Lc를 넘으면 '마지막만 길이컷' 규칙 위반
                    ok = False
                    break
                acc += o.cl
            else:
                # 마지막 장
                last_sum = acc + o.cl
                if last_sum < Lc:
                    ok = False
                    break
                length_cut_last = 1 if last_sum > Lc else 0

        if not ok:
            continue

        # 폭컷(각 장 독립) / 자재비 / 컷비
        width_cuts = [1 if o.cw > Wc else 0 for o in oriented_list]
        material_cost = sum(o.panel.price for o in oriented_list)
        total_cuts = sum(width_cuts) + length_cut_last
        cut_cost_total = total_cuts * cut_cost
        total_cost = material_cost + cut_cost_total

        candidates.append(
            Candidate(
                pattern=pattern,
                oriented=oriented_list,
                width_cuts=width_cuts,
                length_cut_last=length_cut_last,
                material_cost=material_cost,
                cut_cost=cut_cost_total,
                total_cost=total_cost,
            )
        )

    # 정렬(총비용 → 총컷수 → 자재비)
    candidates.sort(
        key=lambda c: (
            c.total_cost,
            (sum(c.width_cuts) + c.length_cut_last),
            c.material_cost,
        )
    )
    return candidates


def optimize_rect(
    W: int,
    L: int,
    df_check: pd.DataFrame,
    df_body: pd.DataFrame,
    df_side: pd.DataFrame,
    cut_cost: int,
    mgmt_ratio_pct: float,
) -> Dict[str, Any]:
    # 입력 검증 & 보정
    if L < W:
        return {"status": "error", "message": "길이 L ≥ 폭 W 조건이 필요합니다."}
    if W > 1900:
        return {"status": "error", "message": "폭 W ≤ 1900 제한을 초과했습니다."}
    Wc, Lc = int(W + 100), int(L + 100)

    bodies = df_to_panels(df_body, "B")
    sides = df_to_panels(df_side, "S")
    if not bodies or not sides:
        return {
            "status": "error",
            "message": "카탈로그에서 바디/사이드 표를 찾지 못했습니다.",
        }

    # 패턴들
    patterns = [
        ["B"],
        ["B", "S"],
        ["B", "S", "S"],
        ["B", "B", "S"],
        ["B", "B", "S", "S"],
    ]

    all_cands: List[Candidate] = []
    for pat in patterns:
        all_cands += eval_pattern_rect(Wc, Lc, pat, bodies, sides, cut_cost)

    if not all_cands:
        return {"status": "no_solution", "message": "구성 가능한 조합이 없습니다."}

    best = all_cands[0]

    # 점검구(바디와 동일 모델) – 첫 번째 B의 이름 사용
    first_body = next((o for o in best.oriented if o.panel.kind == "B"), None)
    check_price, is_double = get_check_price(
        df_check, first_body.panel.name if first_body else ""
    )
    check_total = check_price * (2 if is_double else 1)

    subtotal = best.material_cost + best.cut_cost + check_total
    mgmt_total = int(round(subtotal * (1.0 + mgmt_ratio_pct / 100.0)))

    # 결과 구성
    def row(c: Candidate) -> Dict[str, Any]:
        names = []
        spans = []
        rots = []
        wcuts = []
        for i, o in enumerate(c.oriented):
            names.append(f"{o.panel.name}")
            spans.append(f"{o.cw}×{o.cl}")
            rots.append("90°" if o.rotated else "0°")
            wcuts.append(c.width_cuts[i])
        return {
            "패턴": "+".join(c.pattern),
            "패널명": " + ".join(names),
            "배치치수(cw×cl)": " + ".join(spans),
            "회전": " + ".join(rots),
            "폭컷": sum(wcuts),
            "마지막 길이컷": c.length_cut_last,
            "총컷수": sum(wcuts) + c.length_cut_last,
            "자재비": c.material_cost,
            "절단비": c.cut_cost,
            "총비용": c.total_cost,
        }

    top = [row(c) for c in all_cands[:20]]

    return {
        "status": "ok",
        "mode": "rect",
        "Wc": Wc,
        "Lc": Lc,
        "best": row(best),
        "top": top,
        "detail_best": best,
        "mgmt_total": mgmt_total,
        "subtotal": subtotal,
        "check_price_each": check_price,
        "check_double": is_double,
    }


# =========================================================
# 코너형 최적화 (세면부=B 전용, 샤워부=S 전용)
# =========================================================
def optimize_zone(
    Wc: int, Lc: int, kind: str, bodies: List[Panel], sides: List[Panel], cut_cost: int
) -> Optional[Candidate]:
    """영역 하나(세면부 or 샤워부)에 대해 허용 패턴 전수 후 최소값 선택"""
    if kind == "B":
        patterns = [["B"], ["B", "B"]]
        return (
            eval_pattern_rect(Wc, Lc, patterns[0], bodies, sides, cut_cost)
            + eval_pattern_rect(Wc, Lc, patterns[1], bodies, sides, cut_cost)
            or [None]
        )[0]
    else:
        patterns = [["S"], ["S", "S"]]
        # bodies/sides는 eval 함수 안에서 pattern에 맞게 사용됨
        cands = []
        cands += eval_pattern_rect(Wc, Lc, patterns[0], bodies, sides, cut_cost)
        cands += eval_pattern_rect(Wc, Lc, patterns[1], bodies, sides, cut_cost)
        return cands[0] if cands else None


def optimize_corner(
    S_W: int,
    S_L: int,
    H_W: int,
    H_L: int,
    df_check: pd.DataFrame,
    df_body: pd.DataFrame,
    df_side: pd.DataFrame,
    cut_cost: int,
    mgmt_ratio_pct: float,
) -> Dict[str, Any]:
    # 보정치수
    S_Wc, S_Lc = int(S_W + 100), int(S_L + 100)
    H_Wc, H_Lc = int(H_W + 100), int(H_L + 0)

    if S_Wc <= 0 or S_Lc <= 0 or H_Wc <= 0 or H_Lc < 0:
        return {"status": "error", "message": "치수가 올바르지 않습니다."}
    if S_W < H_W:
        return {
            "status": "error",
            "message": "S_W ≥ H_W 조건(오목부 높이 ≥ 0)이 필요합니다.",
        }

    bodies = df_to_panels(df_body, "B")
    sides = df_to_panels(df_side, "S")
    if not bodies or not sides:
        return {
            "status": "error",
            "message": "카탈로그에서 바디/사이드 표를 찾지 못했습니다.",
        }

    # 영역별 최적
    best_sink = optimize_zone(S_Wc, S_Lc, "B", bodies, sides, cut_cost)
    best_shower = optimize_zone(H_Wc, H_Lc, "S", bodies, sides, cut_cost)
    if (best_sink is None) or (best_shower is None):
        return {
            "status": "no_solution",
            "message": "세면부/샤워부 중 구성 불가 영역이 있습니다.",
        }

    # 합산
    total_material = best_sink.material_cost + best_shower.material_cost
    total_cuts = (sum(best_sink.width_cuts) + best_sink.length_cut_last) + (
        sum(best_shower.width_cuts) + best_shower.length_cut_last
    )
    total_cut_cost = total_cuts * cut_cost
    total_cost = total_material + total_cut_cost

    # 점검구(세면부 첫 바디 기준)
    first_body = next((o for o in best_sink.oriented if o.panel.kind == "B"), None)
    check_price, is_double = get_check_price(
        df_check, first_body.panel.name if first_body else ""
    )
    check_total = check_price * (2 if is_double else 1)

    subtotal = total_material + total_cut_cost + check_total
    mgmt_total = int(round(subtotal * (1.0 + mgmt_ratio_pct / 100.0)))

    def row_zone(c: Candidate, label: str) -> Dict[str, Any]:
        names = " + ".join(o.panel.name for o in c.oriented)
        spans = " + ".join(f"{o.cw}×{o.cl}" for o in c.oriented)
        rots = " + ".join("90°" if o.rotated else "0°" for o in c.oriented)
        wcuts = sum(1 if o.cw > 0 else 0 for o in c.oriented)  # 표시용
        return {
            "영역": label,
            "패턴": "+".join(c.pattern),
            "패널명": names,
            "배치치수(cw×cl)": spans,
            "회전": rots,
            "폭컷": sum(c.width_cuts),
            "마지막 길이컷": c.length_cut_last,
            "총컷수": sum(c.width_cuts) + c.length_cut_last,
            "자재비": c.material_cost,
            "절단비": c.cut_cost,
            "부분합": c.total_cost,
        }

    return {
        "status": "ok",
        "mode": "corner",
        "S_Wc": S_Wc,
        "S_Lc": S_Lc,
        "H_Wc": H_Wc,
        "H_Lc": H_Lc,
        "sink": row_zone(best_sink, "세면부(B)"),
        "shower": row_zone(best_shower, "샤워부(S)"),
        "sum_material": total_material,
        "sum_cut_cost": total_cut_cost,
        "sum_total_cost": total_cost,
        "mgmt_total": mgmt_total,
        "subtotal": subtotal,
        "check_price_each": check_price,
        "check_double": is_double,
    }


# =========================================================
# SVG 렌더링 (사각형 치수 + 패널 윤곽 오버레이 / 코너형 넘버링)
# =========================================================
def svg_arrow(
    x1, y1, x2, y2, label=None, label_pos="mid", stroke="#333", w=1.2, arrow=True
):
    marker = ""
    defs = ""
    if arrow:
        marker = 'marker-end="url(#arrow)" marker-start="url(#arrow)"'
        defs = """
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="#333"/>
          </marker>
        </defs>
        """
    line = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{w}" {marker}/>'
    lbl = ""
    if label:
        if label_pos == "mid":
            tx = (x1 + x2) / 2
            ty = (y1 + y2) / 2 - 6
        elif label_pos == "end":
            tx, ty = x2, y2 - 6
        else:
            tx, ty = x1, y1 - 6
        lbl = f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="11" fill="#111">{label}</text>'
    return defs + line + lbl


def render_rect_with_panels(
    Wc: int, Lc: int, best: Optional[Candidate], title="사각형(보정치수 + 패널윤곽)"
):
    # 스케일
    max_w_px, max_h_px = MAX_RECT_CANVAS_W, MAX_RECT_CANVAS_H
    scale = min(max_w_px / max(Lc, 1), max_h_px / max(Wc, 1))
    W = Lc * scale
    H = Wc * scale
    pad = 24

    outer = f'<rect x="{pad}" y="{pad}" width="{W:.2f}" height="{H:.2f}" fill="none" stroke="#111" stroke-width="1.5"/>'
    y_dim = pad + H + 18
    dim_h = svg_arrow(pad, y_dim, pad + W, y_dim, label=f"Lc = {Lc} mm")
    x_dim = pad - 18
    dim_v = svg_arrow(x_dim, pad + H, x_dim, pad, label=f"Wc = {Wc} mm")

    # 패널 윤곽(빨강=바디, 파랑=사이드), 길이축으로 이어붙임
    overlays = ""
    if best is not None:
        acc = 0
        for i, o in enumerate(best.oriented):
            used_len = o.cl
            if i == len(best.oriented) - 1:
                used_len = max(0, Lc - acc)  # 마지막 장은 남은 길이만큼(컷 반영)
            # 사각형 내부 좌표: x = pad + acc*scale, width = used_len*scale, height = H
            x0 = pad + acc * scale
            y0 = pad
            w = used_len * scale
            h = H
            color = "#e11" if o.panel.kind == "B" else "#06c"
            overlays += f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="none" stroke="{color}" stroke-width="2"/>'
            # 라벨
            name = o.panel.name
            rot = "90°" if o.rotated else "0°"
            overlays += f'<text x="{x0+w/2:.1f}" y="{y0+14:.1f}" text-anchor="middle" font-size="10" fill="{color}">{name} ({rot})</text>'
            acc += o.cl

    lbl = f'<text x="{pad + W/2:.1f}" y="{pad - 8}" text-anchor="middle" font-size="12">{title}</text>'
    svg = f"""
    <svg viewBox="0 0 {W+pad*2:.0f} {H+pad*2+40:.0f}" width="{W+pad*2:.0f}" height="{H+pad*2+40:.0f}" xmlns="http://www.w3.org/2000/svg">
      {outer}{overlays}{dim_h}{dim_v}{lbl}
    </svg>
    """
    components.html(svg, height=int(H + pad * 2 + 60), scrolling=False)


def render_corner_numbered(
    S_W: int, S_L: int, H_W: int, H_L: int, title="코너형(넘버링/치수)"
):
    # 변 길이(원치수 기준)
    s1 = S_L + H_L  # 1번(하단)
    s2 = S_W  # 2번(우측 전체)
    s3 = S_L  # 3번(상단 좌측)
    s4 = S_W - H_W  # 4번(좌측 상단)
    s5 = H_L  # 5번(오목부 하단)
    s6 = H_W  # 6번(우측 상단)
    if s4 < 0:
        components.html(
            '<p style="color:#c00;">오류: S_W ≥ H_W 조건 필요 (오목부 높이가 음수)</p>',
            height=40,
        )
        return

    # 전체 외곽(오목부 포함)
    L_total, W_total = s1, s2

    # 화면 1/3 정도로 스케일
    max_w_px, max_h_px = MAX_RECT_CANVAS_W, MAX_RECT_CANVAS_H
    scale = min(max_w_px / max(L_total, 1), max_h_px / max(W_total, 1))

    # 패딩(↘️ 오버랩 방지 & 우측 살짝 이동)
    pad_left = 60  # ⬅️ 도형을 오른쪽으로
    pad_right = 32
    pad_top = 46  # ⬆️ 상단 여백↑ (제목/3번 라벨 겹침 방지)
    pad_bottom = 52  # ⬇️ 하단 여백↑ (1번 라벨 하단 공간)

    L = L_total * scale
    W = W_total * scale

    # 꼭짓점 좌표 (시계방향 A~F)
    A = (pad_left, pad_top + W)  # 좌하
    B = (pad_left + L, pad_top + W)  # 우하
    C = (pad_left + L, pad_top + W - s6 * scale)  # 우상단 오목 아래
    D = (pad_left + s3 * scale, pad_top + W - s6 * scale)  # 오목 코너
    E = (pad_left + s3 * scale, pad_top)  # 좌상
    F = (pad_left, pad_top)  # 좌상 좌

    # 외곽 폴리라인
    poly = f"""
      <path d="M {A[0]:.2f},{A[1]:.2f} L {B[0]:.2f},{B[1]:.2f} L {C[0]:.2f},{C[1]:.2f}
               L {D[0]:.2f},{D[1]:.2f} L {E[0]:.2f},{E[1]:.2f} L {F[0]:.2f},{F[1]:.2f} Z"
            fill="none" stroke="#111" stroke-width="1.6"/>
    """

    # 라벨 위치(모두 '보이는 쪽'으로 이동: 2·6은 우측 내부, 3은 상단 내부)
    t1 = ((A[0] + B[0]) / 2, A[1] + 18)  # 1: 하단, 바깥 아래
    t2 = (pad_left - 12, pad_top + W / 2)  # 2: 좌측 중
    t3 = ((F[0] + E[0]) / 2, F[1] + 14)  # 3: 상단 좌측, 내부(아래쪽)
    t4 = (E[0] + 8, (E[1] + D[1]) / 2)  # 4: 좌측 상단, 내부
    t5 = ((D[0] + C[0]) / 2, D[1] + 16)  # 5: 오목부 하단, 내부
    t6 = (B[0] - 8, (B[1] + C[1]) / 2)  # 6: 우측 상단, 내부

    labels = f"""
      <text x="{t1[0]:.1f}" y="{t1[1]:.1f}" text-anchor="middle" font-size="11">1: {s1} mm</text>
      <text x="{t2[0]:.1f}" y="{t2[1]:.1f}" text-anchor="end" font-size="11">2:    {s2} mm</text>
      <text x="{t3[0]:.1f}" y="{t3[1]:.1f}" text-anchor="middle" font-size="11">3: {s3} mm</text>
      <text x="{t4[0]:.1f}" y="{t4[1]:.1f}" text-anchor="start"  font-size="11">4: {s4} mm</text>
      <text x="{t5[0]:.1f}" y="{t5[1]:.1f}" text-anchor="middle" font-size="11">5: {s5} mm</text>
      <text x="{t6[0]:.1f}" y="{t6[1]:.1f}" text-anchor="end"    font-size="11">6: {s6} mm</text>
    """

    header = f"""
      <text x="{pad_left + L/2:.1f}" y="{pad_top - 16}" text-anchor="middle" font-size="12">{title}</text>
      <text x="{pad_left + L/2:.1f}" y="{pad_top + W + 34}" text-anchor="middle" font-size="10" fill="#444">
        규칙: 1 = 3 + 5 = {s3} + {s5} = {s1},  2 = 4 + 6 = {s4} + {s6} = {s2}
      </text>
    """

    view_w = L + pad_left + pad_right
    view_h = W + pad_top + pad_bottom

    svg = f"""
    <svg viewBox="0 0 {view_w:.0f} {view_h + 40:.0f}" width="{view_w:.0f}" height="{view_h + 40:.0f}" xmlns="http://www.w3.org/2000/svg">
      {poly}{labels}{header}
    </svg>
    """
    components.html(svg, height=int(view_h + 60), scrolling=False)


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="천장판 최적 조합(치수/넘버링 포함)", layout="wide")
st.title("욕실 천장판 최적 조합 (사각형 / 코너형) • 치수선/넘버링 표시")

with st.sidebar:
    st.header("입력 / 설정")
    mode = st.radio("욕실 형태", ["사각형", "코너형(L자)"])
    st.markdown("---")
    uploaded = st.file_uploader(
        "카탈로그 엑셀 업로드 (시트: '천창판')", type=["xlsx", "xls"]
    )
    use_sample = st.checkbox("업로드 없으면 샘플 DB 사용", value=True)
    cut_cost = st.number_input(
        "절단 1회당 공임 C(원)", min_value=0, value=CUT_COST_DEFAULT, step=500
    )
    mgmt_ratio_pct = st.number_input(
        "관리비율 r(%)", min_value=0.0, value=MGMT_RATIO_DEFAULT, step=0.5
    )

# 카탈로그 로딩
if uploaded is not None:
    try:
        df_check, df_body_raw, df_side_raw = parse_catalog(uploaded)
    except Exception as e:
        st.error(f"엑셀 파싱 실패: {e}")
        st.stop()
else:
    if use_sample:
        df_check, df_body_raw, df_side_raw = sample_catalog()
    else:
        st.info("엑셀을 업로드하거나 '샘플 DB 사용'을 체크하세요.")
        st.stop()

# 미리보기
with st.expander("카탈로그 미리보기", expanded=False):
    st.write("점검구")
    st.dataframe(df_check, use_container_width=True)
    st.write("바디판넬")
    st.dataframe(df_body_raw, use_container_width=True)
    st.write("사이드판넬")
    st.dataframe(df_side_raw, use_container_width=True)

# ===================== 사각형 =====================
if mode == "사각형":
    st.subheader("사각형 입력")
    c1, c2 = st.columns(2)
    with c1:
        W = st.number_input(
            "욕실 폭 W (mm, ≤ 1900)", min_value=200, max_value=1900, value=1350, step=10
        )
    with c2:
        L = st.number_input(
            "욕실 길이 L (mm, L ≥ W)", min_value=int(W), value=2040, step=10
        )

    Wc, Lc = int(W + 100), int(L + 100)
    st.caption(f"보정치수: Wc = {Wc} mm,  Lc = {Lc} mm  (사방 50mm 여유 포함)")

    res = optimize_rect(
        W, L, df_check, df_body_raw, df_side_raw, int(cut_cost), mgmt_ratio_pct
    )
    if res["status"] == "error":
        st.error(res["message"])
        st.stop()
    if res["status"] == "no_solution":
        st.warning(res["message"])
        render_rect_with_panels(Wc, Lc, None, title="사각형(보정치수)")
        st.stop()

    # 도형 렌더: 최선안 오버레이
    best_detail: Candidate = res["detail_best"]
    render_rect_with_panels(
        Wc, Lc, best_detail, title="사각형(보정치수 + 최적 조합 윤곽)"
    )

    st.subheader("최적 조합 (최소 총비용)")
    best = res["best"]
    cols = st.columns(3)
    with cols[0]:
        st.write(f"**패턴**: {best['패턴']}")
        st.write(f"**패널명**: {best['패널명']}")
        st.write(f"**배치치수**: {best['배치치수(cw×cl)']}")
        st.write(f"**회전**: {best['회전']}")
    with cols[1]:
        st.write(
            f"**폭컷**: {best['폭컷']}  |  **마지막 길이컷**: {best['마지막 길이컷']}"
        )
        st.write(f"**총컷수**: {best['총컷수']}")
        st.write(f"**자재비**: {best['자재비']:,}원")
        st.write(f"**절단비**: {best['절단비']:,}원")
        st.write(f"**총비용(자재+절단)**: {best['총비용']:,}원")
    with cols[2]:
        chk_each = res["check_price_each"]
        chk_double = res["check_double"]
        chk_txt = f"{chk_each:,}원" + (" ×2" if chk_double else "")
        st.write(f"**점검구(바디와 동일 모델)**: {chk_txt}")
        st.success(f"**관리비 포함 합계**: {res['mgmt_total']:,}원")
    # ====== 자동저장: 천장 결과를 session_state에 기록 ======
    try:
        st.session_state[CEIL_RESULT_KEY] = {
            "section": "ceil",
            "inputs": {
                "mode": mode,
                "W": int(W), "L": int(L),
                "Wc": int(Wc), "Lc": int(Lc),
                "cut_cost": int(cut_cost),
                "mgmt_ratio_pct": float(mgmt_ratio_pct),
            },
            "result": {
                "status": res.get("status"),
                "message": res.get("message"),
                "best": res.get("best", {}),
                "detail_best": (candidate_to_dict(res["detail_best"]) if "detail_best" in res else {}),
                "summary": res.get("summary", {}),
            },
        }
        st.success("천장 결과 자동저장 완료")
    except Exception as _e:
        st.warning(f"천장 결과 자동저장 중 오류: {_e}")


    st.subheader("상위 후보 (총비용 오름차순)")
    st.dataframe(pd.DataFrame(res["top"]), use_container_width=True)

# ===================== 코너형 =====================
else:
    st.subheader("코너형 입력 (원치수)")
    c1, c2 = st.columns(2)
    with c1:
        S_W = st.number_input("세면부 폭 S_W (mm)", min_value=400, value=1600, step=10)
        S_L = st.number_input(
            "세면부 길이 S_L (mm)", min_value=300, value=1300, step=10
        )
    with c2:
        H_W = st.number_input("샤워부 폭 H_W (mm)", min_value=300, value=1200, step=10)
        H_L = st.number_input("샤워부 길이 H_L (mm)", min_value=300, value=700, step=10)

    S_Wc, S_Lc = int(S_W + 100), int(S_L + 100)
    H_Wc, H_Lc = int(H_W + 100), int(H_L + 0)
    st.caption(
        f"보정치수: 세면부(S) Wc={S_Wc}, Lc={S_Lc} / 샤워부(H) Wc={H_Wc}, Lc={H_Lc}"
    )

    # 넘버링 도형
    render_corner_numbered(S_W, S_L, H_W, H_L, title="코너형(1~6 변 + 길이)")

    res = optimize_corner(
        S_W,
        S_L,
        H_W,
        H_L,
        df_check,
        df_body_raw,
        df_side_raw,
        int(cut_cost),
        mgmt_ratio_pct,
    )
    if res["status"] == "error":
        st.error(res["message"])
        st.stop()
    if res["status"] == "no_solution":
        st.warning(res["message"])
        st.stop()

    st.subheader("영역별 최적 조합")
    st.dataframe(pd.DataFrame([res["sink"], res["shower"]]), use_container_width=True)

    cols = st.columns(3)
    with cols[0]:
        st.write(f"**자재비 합**: {res['sum_material']:,}원")
        st.write(f"**절단비 합**: {res['sum_cut_cost']:,}원")
        st.write(f"**총비용(자재+절단)**: {res['sum_total_cost']:,}원")
    with cols[1]:
        chk_each = res["check_price_each"]
        chk_double = res["check_double"]
        chk_txt = f"{chk_each:,}원" + (" ×2" if chk_double else "")
        st.write(f"**점검구(바디와 동일 모델)**: {chk_txt}")
    with cols[2]:
        st.success(f"**관리비 포함 합계**: {res['mgmt_total']:,}원")
    # ====== 자동저장: 천장 결과를 session_state에 기록 ======
    try:
        st.session_state[CEIL_RESULT_KEY] = {
            "section": "ceil",
            "inputs": {
                "mode": mode,  # "코너형(L자)"
                "S_W": int(S_W), "S_L": int(S_L),
                "H_W": int(H_W), "H_L": int(H_L),
                "S_Wc": int(S_Wc), "S_Lc": int(S_Lc),
                "H_Wc": int(H_Wc), "H_Lc": int(H_Lc),
                "cut_cost": int(cut_cost),
                "mgmt_ratio_pct": float(mgmt_ratio_pct),
            },
            "result": {
                "status": res.get("status"),
                "message": res.get("message"),
                "sink": res.get("sink", {}),       # 세면부 최적안
                "shower": res.get("shower", {}),   # 샤워부 최적안
                "sum_material": res.get("sum_material"),
                "sum_cut_cost": res.get("sum_cut_cost"),
                "sum_total_cost": res.get("sum_total_cost"),
                "check_price_each": res.get("check_price_each"),
                "check_double": res.get("check_double"),
                "subtotal": res.get("subtotal"),
                "mgmt_total": res.get("mgmt_total"),
            },
        }
        st.success("천장 결과 자동저장 완료")
    except Exception as _e:
        st.warning(f"천장 결과 자동저장 중 오류: {_e}")




# ------- 천장 결과 내보내기 -------
st.divider()
st.subheader("천장 결과 내보내기")

def _export_ceil_json():
    data = st.session_state.get(CEIL_RESULT_KEY)
    if not data:
        st.warning("먼저 계산을 실행해 자동저장을 생성하세요.")
        return
    fname = f"ceil_{datetime.now():%Y%m%d_%H%M%S}.json"
    path = os.path.join(EXPORT_DIR, fname)
    _save_json(path, data)
    st.success(f"JSON 내보냈습니다: {path}")

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.button("💾 JSON 내보내기 (파일로 저장)", on_click=_export_ceil_json, key="btn_export_ceil")
with col_e2:
    _data = st.session_state.get(CEIL_RESULT_KEY) or {}
    st.download_button(
        "⬇️ JSON 다운로드 (브라우저)",
        data=json.dumps(_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ceil.json",
        mime="application/json",
        key="btn_download_ceil",
        disabled=not bool(_data),
    )