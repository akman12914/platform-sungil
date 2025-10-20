# -*- coding: utf-8 -*-
# 통합: 천장판 계산 UI + 엔진 + 엑셀 카탈로그 로딩 + 도면/배치행렬 스케치 + 표 + JSON 내보내기
# 역이식: 다운로드 파일 형식 + 인증시스템 + session state + common_styles
# 실행: streamlit run ceil_panel_final2.py

from __future__ import annotations
import io
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict
from collections import Counter, defaultdict
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- Common Styles ---
from common_styles import apply_common_styles, set_page_config

# --- Authentication ---
import auth

# =========================================
# 페이지 설정 및 인증
# =========================================
set_page_config(page_title="천장판 계산 프로그램 (통합)", layout="wide")
apply_common_styles()
auth.require_auth()

# =========================================
# Session State Keys
# =========================================
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

CEIL_DONE_KEY = "ceil_done"
CEIL_RESULT_KEY = "ceil_result"

# =========================================
# 전역 상수/옵션
# =========================================
CUT_COST = 3000
STEP_MM = 50
BODY_MAX_W = 1450
SIDE_MAX_W = 1200


# =========================================
# 공통 유틸
# =========================================
def iround(x: float) -> int:
    return int(math.floor(x + 0.5))


def install_space_rect(W: int, L: int) -> Tuple[int, int]:
    """사각형: 설치공간 보정 (좌우/상하 각 +50)"""
    return int(W) + 100, int(L) + 100


def install_spaces_corner(
    v1: int, v2: int, v3: int, v4: int, v5: int, v6: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """코너형: ((세면 폭,길이), (샤워 폭,길이))"""
    v1, v2, v3, v4, v5, v6 = map(int, (v1, v2, v3, v4, v5, v6))
    sink_w = v2 + 100
    sink_l = (v1 - v5) + 100  # = v3 + 100
    shower_w = v6 + 100
    shower_l = v5 + 50
    return (sink_w, sink_l), (shower_w, shower_l)


def _save_json(path: str, data: dict):
    """JSON 파일 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================
# 카탈로그 모델
# =========================================
@dataclass(frozen=True)
class Panel:
    name: str
    kind: Literal["BODY", "SIDE", "HATCH"]
    w: int
    l: int
    price: int


# 기본 카탈로그(엑셀 업로드 없을 때 사용)
DEFAULT_BODY: List[Panel] = [
    Panel("SI-2", "BODY", 1300, 2000, 24877),
    Panel("SI-3", "BODY", 1300, 1750, 19467),
    Panel("SI-4", "BODY", 1350, 1750, 20465),
    Panel("SI-5", "BODY", 1350, 1750, 22778),
    Panel("SI-6", "BODY", 1450, 1750, 22091),
    Panel("SI-7", "BODY", 1000, 1750, 22305),
    Panel("SI-8", "BODY", 1200, 1750, 25854),
    Panel("SI-9", "BODY", 1200, 2000, 31177),
    Panel("SI-10", "BODY", 1370, 1850, 22091),
    Panel("SI-11", "BODY", 1260, 1850, 21026),
]
DEFAULT_SIDE: List[Panel] = [
    Panel("SIDE-700", "SIDE", 700, 1750, 14110),
    Panel("SIDE-800", "SIDE", 800, 1750, 15954),
    Panel("SIDE-900a", "SIDE", 900, 1750, 18684),
    Panel("SIDE-900b", "SIDE", 900, 960, 10786),  # 회전 후보
    Panel("SIDE-1000", "SIDE", 1000, 1750, 19905),
    Panel("SIDE-1100", "SIDE", 1100, 1850, 20190),
    Panel("SIDE-1200", "SIDE", 1200, 1750, 23454),
    Panel("SIDE-2000x1200", "SIDE", 1200, 2000, 28777),
    Panel("SIDE-750", "SIDE", 750, 1850, 14528),
]
DEFAULT_HATCH: List[Panel] = [
    Panel("SI-2", "HATCH", 700, 900, 8586),
    Panel("SI-3", "HATCH", 700, 900, 8586),
    Panel("SI-4", "HATCH", 700, 900, 8586),
    Panel("SI-5", "HATCH", 500, 650, 6297),
    Panel("SI-6", "HATCH", 700, 900, 8586),
    Panel("SI-7", "HATCH", 450, 450, 4728),
    Panel("SI-8", "HATCH", 450, 450, 4728),
    Panel("SI-9", "HATCH", 450, 450, 4728),
    Panel("SI-10", "HATCH", 650, 900, 8175),
    Panel("SI-11", "HATCH", 750, 900, 8185),
]


def load_catalog_from_excel(
    df: pd.DataFrame,
) -> Tuple[List[Panel], List[Panel], List[Panel]]:
    """
    엑셀 '천장판' 시트 DataFrame → Panel 목록 3종(BODY, SIDE, HATCH).
    예상 컬럼: [판넬/점검구, 품명, 폭, 길이, 소계]
    """
    req_cols = {"판넬/점검구", "품명", "폭", "길이", "소계"}
    if not req_cols.issubset(set(df.columns)):
        raise ValueError(
            f"시트 컬럼이 다릅니다. 필요 컬럼: {req_cols}, 현재: {set(df.columns)}"
        )

    body, side, hatch = [], [], []
    for _, r in df.iterrows():
        kind_raw = str(r["판넬/점검구"]).strip()
        name = str(r["품명"]).strip()
        try:
            w = int(r["폭"])
            l = int(r["길이"])
            price = int(r["소계"])
        except Exception:
            # 사이드 900a/900b 처럼 '품명'이 이름인 경우 폭/길이 숫자 변환 확인
            w = int(float(r["폭"]))
            l = int(float(r["길이"]))
            price = int(float(r["소계"]))
        if "바디" in kind_raw:
            body.append(Panel(name or "NONAME", "BODY", w, l, price))
        elif "사이드" in kind_raw:
            # 품명이 '900a'같이 숫자일 수도 있으니 SIDE- 접두 보정
            name2 = name if name.startswith("SIDE-") else f"SIDE-{name}"
            side.append(Panel(name2, "SIDE", w, l, price))
        else:  # 점검구
            hatch.append(Panel(name, "HATCH", w, l, price))
    return body, side, hatch


# =========================================
# 엔진: 패널 선택/비용
# =========================================
def max_length_capable(catalog: List[Panel], need_w: int) -> int:
    Ls = [p.l for p in catalog if p.w >= need_w]
    return max(Ls) if Ls else 0


def pick_best_panel(
    body_cat: List[Panel],
    side_cat: List[Panel],
    kind: Literal["BODY", "SIDE"],
    need_w: int,
    row_len: int,
    row_idx: int,
    notch: bool = False,
) -> Optional[Tuple[Panel, bool, int, int]]:
    """
    한 셀(행·열)에 들어갈 최저가 패널 선택.
    - 회전 허용: row_idx >= 2 and kind=="SIDE" and SIDE-900b only, need_w in (900,960], row_len <= 900.
    - 길이/폭 초과시 절단 1컷씩 가산.
    - 코너 샤워부는 행마다 notch(2컷) 추가.
    """
    catalog = body_cat if kind == "BODY" else side_cat

    best = None
    # 회전 후보(SIDE-900b → 960×900)
    if kind == "SIDE" and row_idx >= 2 and (900 < need_w <= 960) and (row_len <= 900):
        rot = next((s for s in side_cat if s.name.endswith("900b")), None)
        if rot:
            cuts = (1 if 960 > need_w else 0) + (1 if 900 > row_len else 0)
            extra = 2 if notch else 0
            cost = rot.price + (cuts + extra) * CUT_COST
            best = (rot, True, cuts + extra, cost)

    # 비회전 후보
    for p in catalog:
        if p.w >= need_w and p.l >= row_len:
            cuts = (1 if p.w > need_w else 0) + (1 if p.l > row_len else 0)
            extra = 2 if notch else 0
            cost = p.price + (cuts + extra) * CUT_COST
            cand = (p, False, cuts + extra, cost)
            if (best is None) or (cand[3] < best[3]):
                best = cand
    return best


@dataclass
class RowPlacement:
    zone: str
    kind: Literal["BODY", "SIDE"]
    panel: Panel
    rotated: bool
    need_w: int
    need_l: int
    cuts: int
    cost: int


def fill_vertical_with_edge_align(
    body_cat: List[Panel],
    side_cat: List[Panel],
    width_pattern: List[Tuple[Literal["BODY", "SIDE"], int, str]],
    L_total: int,
    is_corner_shower: bool = False,
) -> Tuple[List[RowPlacement], Optional[str], int, List[int]]:
    """
    width_pattern = [(kind, need_w, zone_label), ...] (가로 열)
    - 같은 행 모든 셀의 길이 동일
    - 1행 길이 = min(잔여 L_total, 각 열이 수용 가능한 최대 길이 cap)
    - 아래 방향(down)으로 반복 적층
    - 코너형 샤워 열은 notch(2컷) 매 행 반영
    반환: (rows, 에러, 총비용, 행길이리스트)
    """
    # 각 열 cap
    caps = []
    for k, w, _ in width_pattern:
        cat = body_cat if k == "BODY" else side_cat
        caps.append(max_length_capable(cat, w))
    if 0 in caps:
        return [], "불가: 해당 폭에서 가능한 패널 없음", 0, []

    rows: List[RowPlacement] = []
    row_lengths: List[int] = []
    total_cost = 0
    L_remain = int(L_total)
    row_idx = 1

    while L_remain > 0:
        row_len = min([L_remain] + caps)
        if row_len <= 0:
            return [], "불가: 세로길이 배치 실패", 0, []
        row_lengths.append(row_len)

        for kind, need_w, zone in width_pattern:
            notch = kind == "SIDE" and is_corner_shower
            pick = pick_best_panel(
                body_cat, side_cat, kind, need_w, row_len, row_idx, notch=notch
            )
            if pick is None:
                return (
                    [],
                    f"불가: {zone} 폭≥{need_w}, 길이≥{row_len} 충족 패널 없음",
                    0,
                    [],
                )
            p, rotated, cuts, cost = pick
            rows.append(
                RowPlacement(
                    f"{zone}/행{row_idx}", kind, p, rotated, need_w, row_len, cuts, cost
                )
            )
            total_cost += cost

        L_remain -= row_len
        row_idx += 1

    return rows, None, total_cost, row_lengths


# =========================================
# 가로 패턴 열거 (사각형)
# =========================================
def enumerate_patterns_rect(
    Wp: int, split: int, mode: Literal["2", "3", "4"], enable_side_bojo: bool = True
):
    """
    사각형 가로 패턴 열거
    - 2판: 기본(BODY=S, SIDE=H), (보조) S>1450이면 SIDE가 경계선을 넘어 세면부 일부 보조(B≤1450, R≤1200, R≥H, B+R=Wp)
    - 3판: (세면2+샤워1) 또는 (세면1+샤워2) — 보조 미적용
    - 4판: (세면1+샤워1) 한정 — BODY≤1450, SIDE≤1200 조건
    """
    S = split + 50  # 세면 요구폭
    H = Wp - S  # 샤워 요구폭
    if S <= 0 or H <= 0:
        return []

    pats: List[List[Tuple[str, int, str]]] = []

    if mode == "4":
        B = min(BODY_MAX_W, S)
        R = Wp - B
        if 0 < R <= SIDE_MAX_W:
            pats.append([("BODY", B, "세면-열1"), ("SIDE", R, "샤워-열1")])
        return pats

    if mode == "2":
        # 기본
        if S <= BODY_MAX_W and H <= SIDE_MAX_W:
            pats.append([("BODY", S, "세면-열1"), ("SIDE", H, "샤워-열1")])
        # 보조 (S>1450)
        if enable_side_bojo and S > BODY_MAX_W:
            R_min = max(H, Wp - BODY_MAX_W)
            R_max = min(SIDE_MAX_W, Wp)
            R_min = ((R_min + STEP_MM - 1) // STEP_MM) * STEP_MM
            R_max = (R_max // STEP_MM) * STEP_MM
            for R in range(R_min, R_max + 1, STEP_MM):
                B = Wp - R
                if 0 < B <= BODY_MAX_W and H <= R <= SIDE_MAX_W:
                    pats.append(
                        [("BODY", B, "세면-열1"), ("SIDE", R, "샤워-열1(보조 포함)")]
                    )
        return pats

    # 3열(세면2+샤워1) 또는 (세면1+샤워2), 보조 미적용
    def gen_cols(total: int, ncols: int, side_label: str):
        out = []
        kinds = ["BODY", "SIDE"]

        def dfs(idx: int, rem: int, acc):
            if idx == ncols:
                if rem == 0:
                    out.append(acc.copy())
                return
            min_rem_need = STEP_MM * (ncols - idx - 1)
            for kind in kinds:
                cap = BODY_MAX_W if kind == "BODY" else SIDE_MAX_W
                w_max = min(cap, rem - min_rem_need)
                w_min = STEP_MM
                if w_max < w_min:
                    continue
                for w in range(w_min, w_max + 1, STEP_MM):
                    acc.append((kind, w, f"{side_label}{idx+1}"))
                    dfs(idx + 1, rem - w, acc)
                    acc.pop()

        dfs(0, total, [])
        return out

    left2 = gen_cols(S, 2, "세면-열")
    right1 = gen_cols(H, 1, "샤워-열")
    left1 = gen_cols(S, 1, "세면-열")
    right2 = gen_cols(H, 2, "샤워-열")
    for lc in left2:
        for rc in right1:
            pats.append(lc + rc)
    for lc in left1:
        for rc in right2:
            pats.append(lc + rc)

    # 중복 제거(라벨 단순화)
    uniq, seen = [], set()
    for p in pats:
        sig = tuple((k, w, z.split("/")[0]) for (k, w, z) in p)
        if sig not in seen:
            seen.add(sig)
            uniq.append(p)
    return uniq


@dataclass
class PatternCost:
    pattern: List[Tuple[str, int, str]]
    rows: List[RowPlacement]
    total_cost: int
    fail_reason: Optional[str] = None
    row_lengths: Optional[List[int]] = None


def cost_of_pattern(
    body_cat: List[Panel],
    side_cat: List[Panel],
    pattern,
    Lp: int,
    is_corner_shower=False,
) -> PatternCost:
    rows, err, tot, rls = fill_vertical_with_edge_align(
        body_cat, side_cat, pattern, Lp, is_corner_shower=is_corner_shower
    )
    if err:
        return PatternCost(pattern, [], 10**12, err, rls)
    return PatternCost(pattern, rows, tot, None, rls)


def search_min_cost_rect(
    body_cat: List[Panel],
    side_cat: List[Panel],
    W: int,
    L: int,
    split: int,
    mode: Literal["2", "3", "4"],
    enable_side_bojo=True,
) -> PatternCost:
    Wp, Lp = install_space_rect(W, L)
    pats = enumerate_patterns_rect(
        Wp, split, mode=mode, enable_side_bojo=enable_side_bojo
    )
    if not pats:
        return PatternCost([], [], 10**12, "가로 패턴 없음", [])
    best: Optional[PatternCost] = None
    for pat in pats:
        pc = cost_of_pattern(body_cat, side_cat, pat, Lp, is_corner_shower=False)
        if pc.fail_reason:
            continue
        if (best is None) or (pc.total_cost < best.total_cost):
            best = pc
    return best if best else PatternCost([], [], 10**12, "모든 패턴 불가", [])


def search_min_cost_rect_global(
    body_cat: List[Panel],
    side_cat: List[Panel],
    W: int,
    L: int,
    split: int,
    enable_side_bojo=True,
):
    cands = []
    for m in ["2", "3", "4"]:
        cands.append(
            (
                m,
                search_min_cost_rect(
                    body_cat,
                    side_cat,
                    W,
                    L,
                    split,
                    mode=m,
                    enable_side_bojo=enable_side_bojo,
                ),
            )
        )
    m_best, pc_best = min(cands, key=lambda x: x[1].total_cost if x[1].rows else 10**12)
    return m_best, pc_best


def search_min_cost_corner_joint(
    body_cat: List[Panel],
    side_cat: List[Panel],
    v1: int,
    v2: int,
    v3: int,
    v4: int,
    v5: int,
    v6: int,
    allow_side_bojo: bool = True,
) -> PatternCost:
    (sw, sl), (ww, wl) = install_spaces_corner(v1, v2, v3, v4, v5, v6)
    total_Wp = sw + ww
    patterns: List[List[Tuple[str, int, str]]] = []

    if sw <= BODY_MAX_W and ww <= SIDE_MAX_W:
        patterns.append([("BODY", sw, "세면"), ("SIDE", ww, "샤워")])

    if allow_side_bojo and (sw > BODY_MAX_W) and (total_Wp <= BODY_MAX_W + SIDE_MAX_W):
        B_min = total_Wp - SIDE_MAX_W
        B_max = min(BODY_MAX_W, total_Wp - ww)
        if B_min <= B_max:
            B_min = ((B_min + STEP_MM - 1) // STEP_MM) * STEP_MM
            B_max = (B_max // STEP_MM) * STEP_MM
            for B in range(B_min, B_max + 1, STEP_MM):
                R = total_Wp - B
                if 0 < B <= BODY_MAX_W and ww <= R <= SIDE_MAX_W:
                    patterns.append(
                        [("BODY", B, "세면"), ("SIDE", R, "샤워(보조 포함)")]
                    )

    best = None
    for pat in patterns:
        pc = cost_of_pattern(body_cat, side_cat, pat, sl, is_corner_shower=True)
        if pc.fail_reason:
            continue
        if (best is None) or (pc.total_cost < best.total_cost):
            best = pc
    return best if best else PatternCost([], [], 10**12, "코너 2열 불가", [])


# =========================================
# 결과 요약 & 요소 테이블
# =========================================
def summarize_solution(
    pc: PatternCost, meta: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """요약표, 요소표, JSON용 기초데이터(개수/단가 합산)"""
    cols = len(pc.pattern)
    rows_len = (len(pc.rows) // cols) if (cols > 0 and pc.rows) else 0

    total_panels = len(pc.rows)
    total_cuts = sum(r.cuts for r in pc.rows)
    total_cost = pc.total_cost
    body_cnt = sum(1 for r in pc.rows if r.kind == "BODY")
    side_cnt = total_panels - body_cnt

    # 크기별(모델별) 개수
    mix_counter = Counter(
        f"{r.panel.name}{'(rot)' if r.rotated else ''} {r.panel.w}x{r.panel.l}"
        for r in pc.rows
    )
    # kind별/규격별 카운트
    kind_size_counter = defaultdict(int)
    for r in pc.rows:
        k = f"{r.kind}:{r.panel.w}x{r.panel.l}"
        kind_size_counter[k] += 1

    # 요약 DF
    summary_dict = {
        **meta,
        "배치행렬": f"{rows_len}x{cols}" if pc.rows else "-",
        "총판넬수": total_panels,
        "바디개수": body_cnt,
        "사이드개수": side_cnt,
        "크기별개수": dict(mix_counter),
        "총절단수": total_cuts,
        "총단가합계": total_cost,
        "실패사유": pc.fail_reason or "",
    }
    df_summary = pd.DataFrame([summary_dict])

    # 요소 DF
    rows_out = []
    # 행 길이 목록 → 행번호/길이 표기(배치행렬 스케치에도 사용)
    row_lengths = pc.row_lengths or []
    row_len_map = {i + 1: L for i, L in enumerate(row_lengths)}

    # 열 폭(need_w) 시그니처(패턴으로부터)
    col_widths = [w for _, w, _ in pc.pattern]

    # rows를 행 단위로 보기 좋게
    if pc.rows:
        cols_n = len(pc.pattern)
        for i, r in enumerate(pc.rows):
            # 행/열 번호
            row_idx = (i // cols_n) + 1
            col_idx = (i % cols_n) + 1
            rows_out.append(
                {
                    "행": row_idx,
                    "열": col_idx,
                    "zone": r.zone,
                    "kind": r.kind,
                    "model": r.panel.name + ("(rot)" if r.rotated else ""),
                    "need_w": r.need_w,
                    "need_l": r.need_l,
                    "panel_w": r.panel.w,
                    "panel_l": r.panel.l,
                    "cuts": r.cuts,
                    "unit_price": r.panel.price,
                    "cell_cost": r.cost,
                }
            )
    df_elements = pd.DataFrame(rows_out)

    # JSON 기본 파츠: kind별/규격별 개수, 총단가
    json_parts = {
        "총개수": int(total_panels),
        "총절단": int(total_cuts),
        "총단가": int(total_cost),
        "kind_size_counts": dict(kind_size_counter),
        "row_lengths": row_lengths,
        "col_widths": col_widths,
    }
    return df_summary, df_elements, json_parts


# =========================================
# Pillow 폰트 로딩
# =========================================
def _get_font(size: int = 16) -> Optional[ImageFont.FreeTypeFont]:
    """한글 폰트 로딩 (NanumGothic.ttf → 시스템 폰트 → 기본)"""
    try:
        return ImageFont.truetype("NanumGothic.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("malgun.ttf", size)  # Windows
        except Exception:
            try:
                return ImageFont.truetype(
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size
                )  # Linux
            except Exception:
                return ImageFont.load_default()


# =========================================
# 도면 그리기 (평면도)
# =========================================
def draw_rect_plan(
    W: int, L: int, split: Optional[int] = None, canvas_w: int = 760, margin: int = 20
) -> Image.Image:
    CANVAS_W = int(canvas_w)
    MARGIN = int(margin)
    sx = (CANVAS_W - 2 * MARGIN) / max(1.0, float(W))
    sy = sx
    CANVAS_H = int(L * sy + 2 * MARGIN)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = MARGIN, MARGIN
    x1 = x0 + int(W * sx)
    y1 = y0 + int(L * sy)
    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    if split is not None:
        gx = x0 + int(split * sx)
        drw.line([gx, y0, gx, y1], fill="blue", width=3)

        # 라벨 추가
        font = _get_font(14)
        drw.text((x0 + split * sx // 2, y0 + 10), "세면부", fill="darkblue", font=font)
        drw.text(
            (x0 + split * sx + (x1 - gx) // 2 - 20, y0 + 10),
            "샤워부",
            fill="darkblue",
            font=font,
        )

    return img


def draw_corner_plan(
    v1: int,
    v2: int,
    v3: int,
    v4: int,
    v5: int,
    v6: int,
    split_on: bool = False,
    canvas_w: int = 760,
    margin: int = 20,
) -> Image.Image:
    CANVAS_W = int(canvas_w)
    MARGIN = int(margin)
    sx = (CANVAS_W - 2 * MARGIN) / max(1.0, float(v1))
    sy = sx
    CANVAS_H = int(v2 * sy + 2 * MARGIN)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = MARGIN, MARGIN

    def X(mm):
        return int(round(x0 + mm * sx))

    def Y(mm):
        return int(round(y0 + mm * sy))

    drw.rectangle([X(0), Y(0), X(v1), Y(v2)], outline="black", width=3)
    notch_x0, notch_x1 = v1 - v5, v1
    notch_y0, notch_y1 = 0, v6
    drw.rectangle(
        [X(notch_x0), Y(notch_y0), X(notch_x1), Y(notch_y1)],
        fill="white",
        outline="white",
    )
    drw.line([X(notch_x0), Y(0), X(notch_x0), Y(v6)], fill="black", width=3)
    drw.line([X(notch_x0), Y(v6), X(v1), Y(v6)], fill="black", width=3)

    if split_on:
        drw.line([X(v3), Y(0), X(v3), Y(v2)], fill="blue", width=3)

        # 라벨 추가
        font = _get_font(14)
        drw.text((X(v3 // 2), Y(v2 // 2)), "세면부", fill="darkblue", font=font)
        drw.text((X(v3 + v5 // 2), Y(v6 // 2)), "샤워부", fill="darkblue", font=font)

    return img


# =========================================
# 배치행렬 스케치(셀 좌표)
# =========================================
def matrix_layout_coords(col_widths_mm: List[int], row_heights_mm: List[int]):
    cols = len(col_widths_mm)
    rows = len(row_heights_mm)
    x_edges = [0]
    for w in col_widths_mm:
        x_edges.append(x_edges[-1] + int(w))
    y_edges = [0]
    for h in row_heights_mm:
        y_edges.append(y_edges[-1] + int(h))

    cells = []
    for r in range(rows):  # r=0 아래행
        for c in range(cols):
            x0 = x_edges[c]
            x1 = x_edges[c + 1]
            y0 = y_edges[r]
            y1 = y_edges[r + 1]
            cells.append(
                {
                    "row": r + 1,
                    "col": c + 1,
                    "x0_mm": x0,
                    "y0_mm": y0,  # bottom-left
                    "x1_mm": x1,
                    "y1_mm": y1,  # top-right
                    "w_mm": x1 - x0,
                    "h_mm": y1 - y0,
                }
            )
    return cells, (x_edges[-1], y_edges[-1])


def draw_matrix_sketch(
    col_widths_mm: List[int],
    row_heights_mm: List[int],
    cell_labels: Optional[Dict[Tuple[int, int], str]] = None,
    scale: float = 0.2,
    margin_px: int = 20,
) -> Image.Image:
    cells, (Wmm, Lmm) = matrix_layout_coords(col_widths_mm, row_heights_mm)
    img_w = int(Wmm * scale) + margin_px * 2
    img_h = int(Lmm * scale) + margin_px * 2
    img = Image.new("RGB", (max(600, img_w), max(360, img_h)), "white")
    draw = ImageDraw.Draw(img)
    x0 = margin_px
    y0 = margin_px
    x1 = x0 + int(Wmm * scale)
    y1 = y0 + int(Lmm * scale)
    draw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    font = _get_font(11)

    for cell in cells:
        cx0 = x0 + int(cell["x0_mm"] * scale)
        cx1 = x0 + int(cell["x1_mm"] * scale)
        cy1 = y1 - int(cell["y0_mm"] * scale)
        cy0 = y1 - int(cell["y1_mm"] * scale)
        draw.rectangle([cx0, cy0, cx1, cy1], outline="#666666", width=2)
        label = (
            cell_labels.get((cell["row"], cell["col"]), "")
            if cell_labels
            else f"R{cell['row']}-C{cell['col']}"
        )
        tx = (cx0 + cx1) // 2 - 32
        ty = (cy0 + cy1) // 2 - 10

        # 멀티라인 텍스트 처리
        lines = label.split("\n")
        for i, line in enumerate(lines):
            draw.text((tx, ty + i * 14), line, fill="black", font=font)

    return img


# =========================================
# UI 시작
# =========================================
st.title("천장판 계산 프로그램 (UI + 엔진 통합)")

# -------- 카탈로그 업로드 --------
with st.sidebar:
    st.header("① 천장판 데이터 로딩")
    up = st.file_uploader("엑셀 업로드 (시트명: '천장판')", type=["xlsx"])
    material = st.selectbox("재질", ["GRP", "FRP", "기타"], index=0)
    st.caption("미업로드 시 기본 카탈로그 사용")

if up:
    try:
        xls = pd.ExcelFile(up)
        df_cat = pd.read_excel(xls, sheet_name="천장판")
        BODY, SIDE, HATCH = load_catalog_from_excel(df_cat)
        st.success(
            f"카탈로그 로드 완료 — BODY {len(BODY)}종, SIDE {len(SIDE)}종, 점검구 {len(HATCH)}종"
        )
    except Exception as e:
        st.error(f"엑셀 파싱 실패: {e}")
        BODY, SIDE, HATCH = DEFAULT_BODY, DEFAULT_SIDE, DEFAULT_HATCH
else:
    BODY, SIDE, HATCH = DEFAULT_BODY, DEFAULT_SIDE, DEFAULT_HATCH

# 카탈로그 확인 UI (Expander)
with st.expander("📋 카탈로그 확인 (업로드 데이터)", expanded=False):
    st.markdown("### 점검구 카탈로그")
    df_check_display = pd.DataFrame(
        [{"이름": h.name, "폭": h.w, "길이": h.l, "가격": h.price} for h in HATCH]
    )
    st.dataframe(df_check_display, use_container_width=True)
    st.caption(f"총 {len(HATCH)}개 항목")

    st.markdown("### 바디판넬 카탈로그")
    df_body_display = pd.DataFrame(
        [{"이름": b.name, "폭": b.w, "길이": b.l, "가격": b.price} for b in BODY]
    )
    st.dataframe(df_body_display, use_container_width=True)
    st.caption(f"총 {len(BODY)}개 항목")

    st.markdown("### 사이드판넬 카탈로그")
    df_side_display = pd.DataFrame(
        [{"이름": s.name, "폭": s.w, "길이": s.l, "가격": s.price} for s in SIDE]
    )
    st.dataframe(df_side_display, use_container_width=True)
    st.caption(f"총 {len(SIDE)}개 항목")

    # 통계 요약
    st.markdown("---")
    st.markdown("#### 📊 카탈로그 통계")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("점검구", f"{len(HATCH)}종")
    with col2:
        st.metric("바디판넬", f"{len(BODY)}종")
    with col3:
        st.metric("사이드판넬", f"{len(SIDE)}종")

# -------- 입력 ----------
st.header("② 사용자 입력")
bath_type = st.radio("욕실유형", ["사각형 욕실", "코너형 욕실"], horizontal=True)
st.markdown(
    "> 설치공간 보정: 사각 W′=W+100, L′=L+100 / 코너 (세면: 폭=2+100, 길이=(1−5)+100), (샤워: 폭=6+100, 길이=5+50)"
)

calc_btn = None
if bath_type == "사각형 욕실":
    c1, c2, c3 = st.columns(3)
    with c1:
        W = st.number_input("가로 W (mm)", min_value=500, value=2000, step=50)
    with c2:
        L = st.number_input("세로 L (mm)", min_value=500, value=1600, step=50)
    with c3:
        split_on = st.radio("세면/샤워 경계선", ["없음", "있음"], horizontal=True)
    split = None
    if split_on == "있음":
        split = st.slider(
            "경계선 X (mm, 가로 기준)",
            min_value=100,
            max_value=int(W),
            step=50,
            value=min(900, int(W)),
        )

    # 평면도
    st.subheader("도면 미리보기 — 사각")
    st.image(draw_rect_plan(W, L, split), use_container_width=False)

    # 계산 옵션
    opt_col = st.columns(3)
    with opt_col[0]:
        side_bojo = st.checkbox("2판 모드: 사이드 보조 커버 허용", value=True)
    with opt_col[1]:
        mode_force = st.selectbox(
            "가로 모드", ["최소단가 자동(2/3/4)", "2", "3", "4"], index=0
        )
    with opt_col[2]:
        hatch_model = st.selectbox(
            "점검구(선택)", ["없음"] + [h.name for h in HATCH], index=0
        )

    calc_btn = st.button("계산 실행", type="primary")

else:
    colA, colB = st.columns(2)
    with colA:
        v3 = st.number_input("3번 변 (mm)", min_value=100, value=800, step=50)
        v5 = st.number_input(
            "5번 변 (오목 가로, mm)", min_value=100, value=900, step=50
        )
        v1 = int(v3 + v5)
        st.text_input("1번 = 3+5", value=str(v1), disabled=True)
    with colB:
        v4 = st.number_input(
            "4번 변 (오목 세로, mm)", min_value=100, value=600, step=50
        )
        v6 = st.number_input("6번 변 (mm)", min_value=100, value=900, step=50)
        v2 = int(v4 + v6)
        st.text_input("2번 = 4+6", value=str(v2), disabled=True)

    st.subheader("도면 미리보기 — 코너")
    st.image(
        draw_corner_plan(v1, v2, v3, v4, v5, v6, split_on=True),
        use_container_width=False,
    )

    opt_col = st.columns(3)
    with opt_col[0]:
        side_bojo = st.checkbox("2판 모드: 사이드 보조 커버 허용", value=True)
    with opt_col[1]:
        hatch_model = st.selectbox(
            "점검구(선택)", ["없음"] + [h.name for h in HATCH], index=0
        )
    with opt_col[2]:
        st.write(
            "세로 적층: 항상 아래 방향, 1행 회전 금지, 2행부터 SIDE-900b 회전 절감 조건 적용"
        )

    calc_btn = st.button("계산 실행", type="primary")

# =========================================
# 계산 실행
# =========================================
if calc_btn:
    try:
        if bath_type == "사각형 욕실":
            if split is None:
                split = max(100, W // 2)  # 경계 없으면 임시 중앙 분할 유도

            # 모드별 탐색
            if mode_force == "최소단가 자동(2/3/4)":
                mode, pc = search_min_cost_rect_global(
                    BODY, SIDE, W, L, split, enable_side_bojo=side_bojo
                )
            else:
                mode = mode_force
                pc = search_min_cost_rect(
                    BODY, SIDE, W, L, split, mode=mode, enable_side_bojo=side_bojo
                )

            Wp, Lp = install_space_rect(W, L)
            meta = {
                "유형": "사각",
                "입력치수": f"W={W}, L={L}, split={split}",
                "설치공간": f"W′={Wp}, L′={Lp}",
                "선택모드": mode,
            }
        else:
            pc = search_min_cost_corner_joint(
                BODY, SIDE, v1, v2, v3, v4, v5, v6, allow_side_bojo=side_bojo
            )
            (sw, sl), (ww, wl) = install_spaces_corner(v1, v2, v3, v4, v5, v6)
            meta = {
                "유형": "코너",
                "입력치수": f"1={v1},2={v2},3={v3},4={v4},5={v5},6={v6}",
                "설치공간": f"세면 {sw}×{sl}, 샤워 {ww}×{wl}(세로목표 {sl})",
                "선택모드": "2(조인트)",
            }

        # 요약/요소표
        df_summary, df_elements, json_parts = summarize_solution(pc, meta)

        # -------- 배치행렬 스케치 --------
        col_widths = json_parts.get("col_widths", [])
        row_lengths = json_parts.get("row_lengths", [])
        cell_labels = {}
        if not df_elements.empty and col_widths and row_lengths:
            # 요소 테이블 기반 라벨: R행-C열\n모델명
            cols_n = len(col_widths)
            for i, row in df_elements.iterrows():
                r = int(row["행"])
                c = int(row["열"])
                label = f"R{r}-C{c}\n{row['model']}"
                cell_labels[(r, c)] = label
            sketch = draw_matrix_sketch(
                col_widths, row_lengths, cell_labels=cell_labels, scale=0.22
            )
            st.subheader("배치행렬 스케치")
            st.image(
                sketch,
                caption=f"행렬 {len(row_lengths)}×{len(col_widths)}",
                use_container_width=False,
            )

        # -------- 표(요약/상세) --------
        st.subheader("요약")
        st.dataframe(df_summary, use_container_width=True)

        st.subheader("요소(셀별 패널/절단/비용)")
        st.dataframe(df_elements, use_container_width=True)

        # -------- 크기별/종류별 집계표 --------
        if not df_elements.empty:
            g_kind = (
                df_elements.assign(
                    dim=lambda d: d["panel_w"].astype(int).astype(str)
                    + "x"
                    + d["panel_l"].astype(int).astype(str)
                )
                .groupby(["kind", "dim"])
                .size()
                .reset_index(name="개수")
                .rename(columns={"dim": "치수"})
            )

            st.subheader("종류·규격별 개수")
            st.dataframe(g_kind, use_container_width=True)

        # -------- 점검구 선택 반영 --------
        hatch_count = 0
        hatch_price = 0
        hatch_name = None
        if hatch_model and hatch_model != "없음":
            sel = next((h for h in HATCH if h.name == hatch_model), None)
            if sel:
                hatch_count = 1
                hatch_price = sel.price
                hatch_name = sel.name
                st.info(
                    f"점검구 선택: {hatch_name} ({sel.w}x{sel.l}) — {sel.price:,}원"
                )

        # -------- JSON 내보내기 --------
        body_models = Counter([r.panel.name for r in pc.rows if r.kind == "BODY"])
        side_models = Counter([r.panel.name for r in pc.rows if r.kind == "SIDE"])
        body_top = (None, 0)
        side_top = (None, 0)
        if body_models:
            body_top = max(body_models.items(), key=lambda x: x[1])
        if side_models:
            side_top = max(side_models.items(), key=lambda x: x[1])

        export_json = {
            "재질": material,
            "바디판넬": {"종류": body_top[0] or "", "개수": int(body_top[1])},
            "사이드판넬": {"종류": side_top[0] or "", "개수": int(side_top[1])},
            "총개수": int(df_summary.at[0, "총판넬수"]) if not df_summary.empty else 0,
            "점검구": int(hatch_count),
            "단가": (
                int(df_summary.at[0, "총단가합계"]) + int(hatch_price)
                if not df_summary.empty
                else 0
            ),
        }
        st.subheader("JSON 미리보기")
        st.code(json.dumps(export_json, ensure_ascii=False, indent=2), language="json")

        # 다운로드 버튼
        buf = io.BytesIO(
            json.dumps(export_json, ensure_ascii=False, indent=2).encode("utf-8")
        )
        st.download_button(
            "JSON 다운로드",
            data=buf,
            file_name="ceiling_panels_order.json",
            mime="application/json",
        )

        # ====== Session State 자동저장 ======
        try:
            st.session_state[CEIL_RESULT_KEY] = {
                "section": "ceil",
                "inputs": {
                    "bath_type": bath_type,
                    "material": material,
                    **meta,
                },
                "result": {
                    "pattern_cost": {
                        "pattern": pc.pattern,
                        "total_cost": pc.total_cost,
                        "fail_reason": pc.fail_reason,
                        "row_lengths": pc.row_lengths,
                    },
                    "summary": (
                        df_summary.to_dict("records")[0] if not df_summary.empty else {}
                    ),
                    "elements": (
                        df_elements.to_dict("records") if not df_elements.empty else []
                    ),
                    "json_export": export_json,
                },
            }
            st.session_state[CEIL_DONE_KEY] = True

            # JSON 파일 자동 저장 (exports 폴더)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"ceil_{timestamp}.json"
            json_path = os.path.join(EXPORT_DIR, json_filename)
            _save_json(json_path, st.session_state[CEIL_RESULT_KEY])

            st.success(f"✅ 천장 결과 자동저장 완료 (Session State + {json_filename})")
        except Exception as save_err:
            st.warning(f"⚠️ 자동저장 중 오류: {save_err}")

    except Exception as e:
        st.error(f"계산 실패: {e}")
        import traceback

        st.code(traceback.format_exc())
