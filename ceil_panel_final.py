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

# 공유 카탈로그 세션 키 (모든 페이지에서 공통 사용)
SHARED_EXCEL_KEY = "shared_excel_file"
SHARED_EXCEL_NAME_KEY = "shared_excel_filename"

# 공유 욕실 정보 세션 키 (바닥판에서 입력, 벽판/천장판에서 사용)
SHARED_BATH_SHAPE_KEY = "shared_bath_shape"  # 욕실 형태: "사각형" or "코너형"
SHARED_BATH_WIDTH_KEY = "shared_bath_width"  # 욕실 폭 (W)
SHARED_BATH_LENGTH_KEY = "shared_bath_length"  # 욕실 길이 (L)
SHARED_SINK_WIDTH_KEY = "shared_sink_width"  # 세면부 폭 (경계선 정보, split용)
SHARED_MATERIAL_KEY = "shared_floor_material"  # 바닥판 재료

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
    """사각형: 설치공간 보정 (좌우/상하 각 +25)"""
    return int(W) + 50, int(L) + 50


def install_spaces_corner(
    v1: int, v2: int, v3: int, v4: int, v5: int, v6: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """코너형: ((세면 폭,길이), (샤워 폭,길이)) - 각 변 25로 보정"""
    v1, v2, v3, v4, v5, v6 = map(int, (v1, v2, v3, v4, v5, v6))
    sink_w = v2 + 50
    sink_l = (v1 - v5) + 50  # = v3 + 50
    shower_w = v6 + 50
    shower_l = v5 + 25
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
# 공동 보조(조인트) 허용 : 세면부의 일부를 사이드 판넬로 커버
# =========================================
def enumerate_joint_assist_patterns(total_Wp:int,
                                    body_max:int,
                                    side_max:int,
                                    ww_min:int,
                                    step:int=STEP_MM,
                                    sink_label="세면-열1",
                                    shower_label="샤워-열1(보조 포함)"):
    pats = []
    B_min = total_Wp - side_max
    B_max = min(body_max, total_Wp - ww_min)
    if B_min > B_max:
        return pats

    def ceil_step(x):  return ((x + step - 1) // step) * step
    def floor_step(x): return (x // step) * step

    B_min = ceil_step(B_min)
    B_max = floor_step(B_max)

    for B in range(B_min, B_max + 1, step):
        R = total_Wp - B
        if ww_min <= R <= side_max:
            pats.append([("BODY", B, sink_label), ("SIDE", R, shower_label)])
    return pats

# =========================================
# 가로 패턴 열거 (자동: BODY 우선, 필요 시 SIDE 보조)
# =========================================
def enumerate_patterns_rect(
    Wp: int,
    split: int,
    enable_side_bojo: bool = True,
    require_body: bool = True,
) -> List[List[Tuple[str, int, str]]]:
    """
    자동 패턴(2열 고정):
      - 원칙: 세면부는 BODY 우선 설치(필수). 잔여 폭은 SIDE가 담당.
      - S = split + 25 (세면 요구폭), H = Wp - S (샤워 요구폭)
      - 기본: (BODY=S, SIDE=H)이 규격 내이면 채택
      - S > BODY_MAX_W 인 경우: '조인트 보조허용' 방식으로 B를 줄이고 SIDE가 일부 세면을 보조
        (B + R = Wp,  R 은 최소 H 이상, R ≤ SIDE_MAX_W)
    """
    S = split + 25   # 세면 요구폭
    H = Wp - S       # 샤워 요구폭
    if S <= 0 or H <= 0:
        return []

    patterns: List[List[Tuple[str, int, str]]] = []

    # 1) 기본 규칙: BODY=S, SIDE=H 가 각자 최대폭 이내면 그대로 사용
    if require_body and (S <= BODY_MAX_W) and (H <= SIDE_MAX_W):
        patterns.append([("BODY", S, "세면-열1"), ("SIDE", H, "샤워-열1")])

    # 2) 세면 요구폭이 BODY 한계를 넘는 경우 → 조인트 보조 허용으로 통일
    #    BODY 폭 B를 줄이고 SIDE가 일부를 보조하여 총합이 Wp가 되게 함
    if enable_side_bojo and (S > BODY_MAX_W):
        patterns.extend(
            enumerate_joint_assist_patterns(
                total_Wp=Wp,
                body_max=BODY_MAX_W,
                side_max=SIDE_MAX_W,
                ww_min=H,                         # 샤워 구역이 최소 확보해야 하는 폭
                step=STEP_MM,
                sink_label="세면-열1",
                shower_label="샤워-열1(보조 포함)",
            )
        )

    # 중복 제거 (같은 (kind, width) 조합이면 1개만 남김)
    uniq, seen = [], set()
    for p in patterns:
        sig = tuple((k, w) for (k, w, _z) in p)
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


# =========================================
# 최소단가 탐색 (자동, 모드 제거)
# =========================================
def search_min_cost_rect(
    body_cat: List[Panel],
    side_cat: List[Panel],
    W: int,
    L: int,
    split: int,
    enable_side_bojo: bool = True,
) -> PatternCost:
    """
    - install_space_rect 로 보정치수(W′, L′) 계산
    - enumerate_patterns_rect 로 후보 패턴 열거(항상 2열: BODY + SIDE)
    - 각 패턴에 대해 세로 적층(행) 배치 비용을 계산 → 최소 비용 선택
    """
    Wp, Lp = install_space_rect(W, L)
    pats = enumerate_patterns_rect(Wp, split, enable_side_bojo=enable_side_bojo, require_body=True)
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
            rows_out.append({
                "행": row_idx, "열": col_idx,
                "zone": r.zone,
                "kind": r.kind,
                "품명": r.panel.name + ("(rot)" if r.rotated else ""),
                "설치폭": r.need_w, "설치길이": r.need_l,
                "판넬폭": r.panel.w, "판넬길이": r.panel.l,
                "절단횟수": r.cuts, "판넬소계": r.panel.price,
                "절단시공비포함 판넬소계": r.cost,
            })
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
    W: int, L: int, split: Optional[int] = None,
    canvas_w: int = 760, canvas_h: int = 540, margin: int = 20
) -> Image.Image:
    CANVAS_W = int(canvas_w)
    CANVAS_H = int(canvas_h)
    MARGIN   = int(margin)

    sx = (CANVAS_W - 2*MARGIN) / max(1.0, float(W))
    sy = (CANVAS_H - 2*MARGIN) / max(1.0, float(L))
    s  = min(sx, sy)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = MARGIN, MARGIN
    x1 = x0 + int(W * s)
    y1 = y0 + int(L * s)

    dx = (CANVAS_W - 2*MARGIN - int(W * s)) // 2
    dy = (CANVAS_H - 2*MARGIN - int(L * s)) // 2
    x0 += dx; x1 += dx
    y0 += dy; y1 += dy

    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)
    if split is not None:
        gx = x0 + int(split * s)
        drw.line([gx, y0, gx, y1], fill="blue", width=3)
    return img


def draw_corner_plan(
    v1: int, v2: int, v3: int, v4: int, v5: int, v6: int,
    split_on: bool=False,
    show_shower_label: bool=False,
    canvas_w: int=760, canvas_h: int=540, margin: int=20
) -> Image.Image:
    CANVAS_W = int(canvas_w)
    CANVAS_H = int(canvas_h)
    MARGIN   = int(margin)

    sx = (CANVAS_W - 2*MARGIN) / max(1.0, float(v1))
    sy = (CANVAS_H - 2*MARGIN) / max(1.0, float(v2))
    s  = min(sx, sy)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    dx = (CANVAS_W - 2*MARGIN - int(v1 * s)) // 2
    dy = (CANVAS_H - 2*MARGIN - int(v2 * s)) // 2
    x0, y0 = MARGIN + dx, MARGIN + dy

    def X(mm): return int(round(x0 + mm * s))
    def Y(mm): return int(round(y0 + mm * s))

    # 외곽
    drw.rectangle([X(0), Y(0), X(v1), Y(v2)], outline="black", width=3)

    # 오목부(상단 우측)
    notch_x0, notch_x1 = v1 - v5, v1
    notch_y0, notch_y1 = 0, v6
    drw.rectangle([X(notch_x0), Y(notch_y0), X(notch_x1), Y(notch_y1)], fill="white", outline="white")
    drw.line([X(notch_x0), Y(notch_y0), X(notch_x0), Y(notch_y1)], fill="black", width=3)

    # 샤워부(하단 우측) - 파란색
    shower_x0, shower_x1 = v1 - v5, v1
    shower_y0, shower_y1 = v2 - v6, v2
    drw.rectangle([X(shower_x0), Y(shower_y0), X(shower_x1), Y(shower_y1)], outline="blue", width=3)
    # 라벨은 기본 숨김 (show_shower_label=True 일 때만 표시)
    if show_shower_label:
        try:
            font = ImageFont.load_default()
            drw.text(((X(shower_x0)+X(shower_x1))//2, (Y(shower_y0)+Y(shower_y1))//2),
                     "샤워부", fill="blue", anchor="mm", font=font)
        except TypeError:
            drw.text(( (X(shower_x0)+X(shower_x1))//2 - 20, (Y(shower_y0)+Y(shower_y1))//2 - 8 ),
                     "샤워부", fill="blue")

    if split_on:
        drw.line([X(v3), Y(0), X(v3), Y(v2)], fill="blue", width=3)
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
    st.header("천장판 데이터 로딩")
    st.info("📂 바닥판에서 업로드한 Excel 카탈로그를 사용합니다.")

    # 바닥판에서 공유된 데이터 표시
    shared_shape = st.session_state.get(SHARED_BATH_SHAPE_KEY)
    shared_width = st.session_state.get(SHARED_BATH_WIDTH_KEY)
    shared_length = st.session_state.get(SHARED_BATH_LENGTH_KEY)
    shared_sink_w = st.session_state.get(SHARED_SINK_WIDTH_KEY)

    if shared_shape:
        st.success(f"✅ 바닥판 데이터 사용 중\n- 형태: {shared_shape}\n- 폭×길이: {shared_width}×{shared_length}mm\n- 세면부 폭: {shared_sink_w}mm")

    st.header("욕실유형")
    # 바닥판 데이터가 있으면 자동 설정, 없으면 수동 선택
    if shared_shape:
        bath_type_map = {"사각형": "사각형 욕실", "코너형": "코너형 욕실"}
        bath_type = bath_type_map.get(shared_shape, "사각형 욕실")
        st.radio("욕실유형 (바닥판 자동 반영)", [bath_type], horizontal=False, disabled=True)
    else:
        bath_type = st.radio("욕실유형", ["사각형 욕실", "코너형 욕실"], horizontal=False)

    st.header("계산 옵션 / 관리비율")

    prod_rate_pct = st.number_input("생산관리비율 rₚ (%)",
                                    min_value=0.0, max_value=80.0,
                                    value=20.0, step=0.5, help="예: 20 → 20%")
    sales_rate_pct = st.number_input("영업관리비율 rₛ (%)",
                                     min_value=0.0, max_value=80.0,
                                     value=20.0, step=0.5, help="예: 20 → 20%")

# -------- read Excel file (shared state only) ----------
# 바닥판에서 공유된 Excel 파일 사용
excel_file = st.session_state.get(SHARED_EXCEL_KEY)
excel_filename = st.session_state.get(SHARED_EXCEL_NAME_KEY, "알 수 없음")

if excel_file:
    try:
        xls = pd.ExcelFile(excel_file)
        df_cat = pd.read_excel(xls, sheet_name="천장판")
        BODY, SIDE, HATCH = load_catalog_from_excel(df_cat)

        # 공유 카탈로그 표시
        st.info(f"📂 공유 카탈로그 사용 중: {excel_filename} — BODY {len(BODY)}종, SIDE {len(SIDE)}종, 점검구 {len(HATCH)}종")

        # 👉 시공비 시트에서 천장판 절단 단가 가져오기
        try:
            df_cost = pd.read_excel(xls, sheet_name="시공비")
            df_cost["항목"] = df_cost["항목"].astype(str).str.strip()
            df_cost["공정"] = df_cost["공정"].astype(str).str.strip()

            mask = (df_cost["항목"] == "천장판") & (df_cost["공정"] == "절단")
            if mask.any():
                cut_val = df_cost.loc[mask, "시공비"].iloc[0]
                if isinstance(cut_val, str):
                    cut_val = cut_val.replace(",", "")
                cut_val = float(cut_val)

                # ★ 여기서 그냥 덮어쓰기만 하면 됨
                CUT_COST = int(round(cut_val))

                st.info(f"시공비 시트에서 천장판 절단비 {CUT_COST:,}원 로드됨")
        except Exception as e:
            st.warning(f"'시공비' 시트에서 천장판 절단비를 읽지 못해 기본값({CUT_COST})을 사용합니다. 상세: {e}")

    except Exception as e:
        st.error(f"엑셀 파싱 실패: {e}")
        st.stop()
else:
    st.warning("⚠️ 바닥판 페이지에서 엑셀 파일을 먼저 업로드해주세요.")
    st.info("💡 바닥판에서 업로드한 Excel 카탈로그가 천장판과 벽판에 자동으로 공유됩니다.")
    st.stop()

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

calc_btn = None
if bath_type == "사각형 욕실":
    c1, c2, c3 = st.columns(3)
    with c1:
        # 공유 데이터가 있으면 자동 설정, 없으면 기본값
        default_w = shared_width if shared_width else 2000
        W = st.number_input("가로 W (mm)", min_value=500, value=default_w, step=50,
                           disabled=bool(shared_width),
                           help="바닥판에서 자동 반영" if shared_width else None)
    with c2:
        default_l = shared_length if shared_length else 1600
        L = st.number_input("세로 L (mm)", min_value=500, value=default_l, step=50,
                           disabled=bool(shared_length),
                           help="바닥판에서 자동 반영" if shared_length else None)
    with c3:
        # 공유 경계선 정보가 있으면 자동으로 "있음" 선택
        if shared_sink_w:
            split_on = "있음"
            st.radio("세면/샤워 경계선 (바닥판 자동 반영)", [split_on], horizontal=True, disabled=True)
        else:
            split_on = st.radio("세면/샤워 경계선", ["없음", "있음"], horizontal=True)

    split = None
    if split_on == "있음":
        # 공유 세면부 폭이 있으면 자동 설정
        if shared_sink_w:
            split = shared_sink_w
            st.slider(
                "경계선 X (mm, 가로 기준) - 바닥판 자동 반영",
                min_value=100,
                max_value=int(W),
                step=50,
                value=split,
                disabled=True
            )
        else:
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
    side_bojo = st.checkbox("2판 모드: 사이드 보조 커버 허용", value=True)

    calc_btn = st.button("계산 실행", type="primary")

else:
    # 코너형: 바닥판 치수를 참고값으로 표시
    if shared_width and shared_length:
        st.info(f"ℹ️ 참고: 바닥판 전체 치수 {shared_width}×{shared_length}mm")

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
    st.image(draw_corner_plan(v1,v2,v3,v4,v5,v6, split_on=True, show_shower_label=False), use_container_width=False)

    st.caption("세로 적층: 아래 방향, 1행 회전 금지, 2행부터 SIDE-900b 회전 절감 조건 적용")
    side_bojo = st.checkbox("2판 모드: 사이드 보조 커버 허용", value=True)

    calc_btn = st.button("계산 실행", type="primary")

# =========================================
# 계산 실행 (안전 버전)
# =========================================
if calc_btn:
    try:
        pc = None  # ← 반드시 미리 선언
        meta = {}

        if bath_type == "사각형 욕실":
            # split 보정
            if split is None:
                split = max(100, int(W) // 2)

            # 최소단가(자동, 모드 제거) 계산
            pc = search_min_cost_rect(BODY, SIDE, int(W), int(L), int(split), enable_side_bojo=side_bojo)

            Wp, Lp = install_space_rect(int(W), int(L))
            meta = {
                "유형": "사각",
                "입력치수": f"W={W}, L={L}, split={split}",
                "설치공간": f"W′={Wp}, L′={Lp}",
            }

        else:
            # 코너형
            pc = search_min_cost_corner_joint(BODY, SIDE, int(v1), int(v2), int(v3), int(v4), int(v5), int(v6),
                                              allow_side_bojo=side_bojo)
            (sw, sl), (ww, wl) = install_spaces_corner(int(v1), int(v2), int(v3), int(v4), int(v5), int(v6))
            meta = {
                "유형": "코너",
                "입력치수": f"1={v1},2={v2},3={v3},4={v4},5={v5},6={v6}",
                "설치공간": f"세면 {sw}×{sl}, 샤워 {ww}×{wl}(세로목표 {sl})",
            }

        # pc가 없거나 실패 시 메시지
        if (pc is None) or (pc.fail_reason and not pc.rows):
            st.error(f"계산 실패: {pc.fail_reason if pc else '내부 오류(PC 없음)'}")
            st.stop()

        # 요약/요소표
        df_summary, df_elements, json_parts = summarize_solution(pc, meta)

        # -------- 배치행렬 스케치 (선택) --------
        col_widths = json_parts.get("col_widths", [])
        row_lengths = json_parts.get("row_lengths", [])
        if not df_elements.empty and col_widths and row_lengths:
            cell_labels = {}
            cols_n = len(col_widths)
            for i, row in df_elements.iterrows():
                r = int(row["행"]); c = int(row["열"])
                cell_labels[(r, c)] = f"R{r}-C{c}\n{row['품명']}"   # ← '품명' 사용!
            sketch = draw_matrix_sketch(col_widths, row_lengths, cell_labels=cell_labels, scale=0.22)
            st.subheader("배치행렬 스케치")
            st.image(sketch, caption=f"행렬 {len(row_lengths)}×{len(col_widths)}", use_container_width=False)

        # -------- 표(요약/상세) --------
        st.subheader("요약")
        st.dataframe(df_summary, use_container_width=True)

        st.subheader("요소(셀별 패널/절단/비용)")
        st.dataframe(df_elements, use_container_width=True)

        # -------- 크기별/종류별 집계표 --------
        if not df_elements.empty:
            g_kind = (
                df_elements
                .assign(dim=lambda d: d["판넬폭"].astype(int).astype(str) + "x" + d["판넬길이"].astype(int).astype(str))
                .groupby(["kind", "dim"])
                .size()
                .reset_index(name="개수")
                .rename(columns={"dim": "치수"})
            )
            st.subheader("종류·규격별 개수")
            st.dataframe(g_kind, use_container_width=True)

        # ===============================
        #   🔵 관리비 계산
        # ===============================
        # 소계 합산(절단비 제외)
        body_subtotal = int(df_elements.loc[df_elements["kind"] == "BODY", "판넬소계"].sum()) if not df_elements.empty else 0
        side_subtotal = int(df_elements.loc[df_elements["kind"] == "SIDE", "판넬소계"].sum()) if not df_elements.empty else 0

        # 점검구 자동 매칭: 최다 BODY 품명 → 동일 품명의 HATCH 1개
        hatch_count = 0
        hatch_price = 0
        hatch_name = None
        if not df_elements.empty:
            body_models = Counter([r.panel.name for r in pc.rows if r.kind == "BODY"])
            if body_models:
                body_top_name, _ = max(body_models.items(), key=lambda x: x[1])
                sel_h = next((h for h in HATCH if h.name == body_top_name), None)
                if sel_h:
                    hatch_count = 1
                    hatch_price = sel_h.price
                    hatch_name = sel_h.name

        hatch_subtotal = int(hatch_price * hatch_count)

        subtotal_sum = int(body_subtotal + side_subtotal + hatch_subtotal)

        # 비율 읽기 (사이드바 입력 그대로 사용)
        rp = float(prod_rate_pct) / 100.0
        rs = float(sales_rate_pct) / 100.0
        if rp >= 1.0 or rs >= 1.0:
            st.error("rₚ, rₛ 는 100% 미만이어야 합니다.")
            st.stop()

        # 생산관리비/영업관리비
        prod_mgmt = (subtotal_sum / (1.0 - rp)) - subtotal_sum if rp > 0 else 0.0
        price_with_prod = subtotal_sum + prod_mgmt

        sales_mgmt = (price_with_prod / (1.0 - rs)) - price_with_prod if rs > 0 else 0.0
        final_price = price_with_prod + sales_mgmt

        st.subheader("관리비/최종단가 계산 결과")
        res_df = pd.DataFrame([{
            "바디 소계": body_subtotal,
            "사이드 소계": side_subtotal,
            "점검구 소계": hatch_subtotal,
            "합계 소계": subtotal_sum,
            "생산관리비": int(round(prod_mgmt)),
            "생산관리비포함 단가": int(round(price_with_prod)),
            "영업관리비": int(round(sales_mgmt)),
            "영업관리비포함 단가(최종)": int(round(final_price)),
            "rₚ(%)": prod_rate_pct,
            "rₛ(%)": sales_rate_pct,
            "자동선정 점검구": (f"{hatch_name}" if hatch_name else "없음"),
        }])
        st.dataframe(res_df, use_container_width=True)

        # -------- JSON 내보내기 --------
        body_models = Counter([r.panel.name for r in pc.rows if r.kind == "BODY"])
        side_models = Counter([r.panel.name for r in pc.rows if r.kind == "SIDE"])
        body_top = max(body_models.items(), key=lambda x: x[1]) if body_models else (None, 0)
        side_top = max(side_models.items(), key=lambda x: x[1]) if side_models else (None, 0)

        export_json = {
            "바디판넬": {"종류": body_top[0] or "", "개수": int(body_top[1])},
            "사이드판넬": {"종류": side_top[0] or "", "개수": int(side_top[1])},
            "점검구": {"종류": hatch_name or "", "개수": int(hatch_count)},
            "총개수": int(df_summary.at[0, "총판넬수"]) if not df_summary.empty else 0,
            "절단포함_총단가": int(df_summary.at[0, "총단가합계"]) if not df_summary.empty else 0,
            "합계소계": int(subtotal_sum),
            "생산관리비율_%": float(prod_rate_pct),
            "생산관리비": int(round(prod_mgmt)),
            "생산관리비포함단가": int(round(price_with_prod)),
            "영업관리비율_%": float(sales_rate_pct),
            "영업관리비": int(round(sales_mgmt)),
            "영업관리비포함단가_최종": int(round(final_price)),
        }

        st.subheader("JSON 미리보기")
        st.code(json.dumps(export_json, ensure_ascii=False, indent=2), language="json")

        buf = io.BytesIO(json.dumps(export_json, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button("JSON 다운로드", data=buf, file_name="ceiling_panels_order.json", mime="application/json")

        # ====== Session State 자동저장 ======
        try:
            st.session_state[CEIL_RESULT_KEY] = {
                "section": "ceil",
                "inputs": {
                    "bath_type": bath_type,
                    "prod_rate_pct": prod_rate_pct,
                    "sales_rate_pct": sales_rate_pct,
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
                    "management_fees": {
                        "subtotal_sum": subtotal_sum,
                        "prod_mgmt": int(round(prod_mgmt)),
                        "sales_mgmt": int(round(sales_mgmt)),
                        "final_price": int(round(final_price)),
                        "hatch_info": {"name": hatch_name, "count": hatch_count, "price": hatch_price},
                    },
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
