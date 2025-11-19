# wall_panel_final.py
# -*- coding: utf-8 -*-
# 벽판 계산 프로그램 (통합) - New Layout Engine + 공통 스타일 + 세션 관리

from __future__ import annotations
import math
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

# --- Common Styles ---
from common_styles import apply_common_styles, set_page_config

# --- Authentication ---
import auth

# =========================================
# Page Configuration
# =========================================
set_page_config(page_title="벽판 계산 프로그램 (통합)", layout="wide")
apply_common_styles()
auth.require_auth()

# =========================================
# Session State Keys
# =========================================
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

FLOOR_DONE_KEY = "floor_done"
FLOOR_RESULT_KEY = "floor_result"
WALL_DONE_KEY = "wall_done"
WALL_RESULT_KEY = "wall_result"
CEIL_DONE_KEY = "ceil_done"
CEIL_RESULT_KEY = "ceil_result"

# 공유 데이터 키
SHARED_EXCEL_KEY = "shared_excel_file"
SHARED_EXCEL_NAME_KEY = "shared_excel_filename"
SHARED_BATH_SHAPE_KEY = "shared_bath_shape"
SHARED_BATH_WIDTH_KEY = "shared_bath_width"
SHARED_BATH_LENGTH_KEY = "shared_bath_length"
SHARED_SINK_WIDTH_KEY = "shared_sink_width"
SHARED_MATERIAL_KEY = "shared_floor_material"

# =========================================
# Utility Functions
# =========================================
def _save_json(path: str, data: dict):
    """JSON 파일 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# 0) 공통 유틸
# =========================================================
def parse_tile(tile_str: str) -> Tuple[int, int]:
    """'300×600' 또는 '250×400' → (300, 600)"""
    a, b = tile_str.replace("x", "×").split("×")
    return int(a), int(b)

def effective_height(H: int, floor_type: str) -> int:
    """바닥판 유형이 PVE면 +50."""
    return int(H) + 50 if floor_type.upper() == "PVE" else int(H)

MIN_EDGE = 80

# =========================================================
# 1) 새 Layout 계산 엔진 (Self-contained)
# =========================================================
def iround(x: float) -> int:
    """Half-up 반올림(ROUND)."""
    return int(math.floor(x + 0.5))

class RuleError(Exception):
    pass

@dataclass
class PanelCell:
    col: int  # 1-based
    row: int  # 1-based (bottom=1)
    w: int
    h: int
    col_tags: Tuple[str, ...] = ()
    row_tags: Tuple[str, ...] = ()
    col_note: str = ""
    row_note: str = ""
    def as_dict(self) -> Dict[str, Any]:
        return {
            "col": self.col, "row": self.row,
            "panel_w": self.w, "panel_h": self.h,
            "col_tags": ",".join(self.col_tags) if self.col_tags else "",
            "row_tags": ",".join(self.row_tags) if self.row_tags else "",
            "col_note": self.col_note,
            "row_note": self.row_note,
        }

def ensure_producible_new(panels: List[PanelCell]):
    for p in panels:
        if p.w <= MIN_EDGE or p.h <= MIN_EDGE:
            raise RuleError(f"PANEL_TOO_SMALL: {p.w}x{p.h} at C{p.col}-R{p.row}")

# --------- 가로(HB/VSTRIP) 분해 ----------
def hb2_split(width: int, TW: int) -> Tuple[int, int, str]:
    """가로발란스(2분할, ROUND)."""
    if width < 2 * TW:
        raise RuleError(f"WIDTH_TOO_SMALL_FOR_BALANCE: {width} < {2*TW}")
    n = iround(width / (2 * TW))
    left = n * TW
    right = width - left
    if left <= MIN_EDGE or right <= MIN_EDGE:
        raise RuleError(f"HB2_BAD_SPLIT: {left}, {right}")
    note = f"HB2: n={n}, left={left}, right={right} (TW={TW})"
    return left, right, note

def split_columns_verbose(W: int, TW: int) -> Tuple[List[Dict], str]:
    """
    columns: [{width, tags(HB/VSTRIP), col_note}, ...]
    horiz_branch: 가로 분해 설명
    규칙: 2400 모듈, 오른쪽 2칸만 HB 허용, 잔여 dW<=80이면 (2400→2400−TW, 세로판→dW+TW)
    """
    cols: List[Dict] = []
    # 80 < W <= 1000 : VSTRIP 1열
    if 80 < W <= 1000:
        cols.append({"width": W, "tags": ("VSTRIP",), "col_note": "W<=1000 → VSTRIP 1열"})
        return cols, "VERTICAL STRIP ONLY (80<W<=1000)"
    # 1000 < W <= 2400 : 단일 열
    if W <= 2400:
        cols.append({"width": W, "tags": tuple(), "col_note": "SINGLE COLUMN (1000<W<=2400)"})
        return cols, "SINGLE COLUMN (1000<W<=2400)"

    # 2400 < W <= 3400 : 2400 + VSTRIP(dW) with ≤80mm correction
    if W <= 3400:
        dW = W - 2400
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400 + VSTRIP(dW), dW={dW}"

    # 3400 < W <= 4800 : HB 2열
    if W <= 4800:
        L, R, note = hb2_split(W, TW)
        cols.append({"width": L, "tags": ("HB",), "col_note": note})
        cols.append({"width": R, "tags": ("HB",), "col_note": note})
        return cols, "HB2 (3400<W<=4800)"

    # 4800 < W <= 5800 : 2400 + 2400 + VSTRIP(dW) with ≤80mm correction
    if W <= 5800:
        dW = W - 4800
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400 + 2400 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400 + 2400 + VSTRIP(dW), dW={dW}"

    # 5800 < W <= 7200 : 2400 + HB2(W-2400)
    if W <= 7200:
        rest = W - 2400
        L, R, note = hb2_split(rest, TW)
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": L, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        cols.append({"width": R, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        return cols, f"2400 + HB2(W-2400), rest={rest}"

    # 7200 < W <= 8200 : 2400×3 + VSTRIP(dW) with ≤80mm correction
    if W <= 8200:
        dW = W - 7200
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400×3 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400×3 + VSTRIP(dW), dW={dW}"

    # 8200 < W <= 9600 : 2400×2 + HB2(W-4800)
    if W <= 9600:
        rest = W - 4800
        L, R, note = hb2_split(rest, TW)
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": L, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        cols.append({"width": R, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        return cols, f"2400×2 + HB2(W-4800), rest={rest}"

    raise RuleError("WIDTH_OUT_OF_RANGE")

# --------- 세로(행) 분해 ----------
def vb_round(H: int, TH: int) -> Tuple[int, int]:
    """세로발란스(ROUND): top=m*TH, bot=H-top."""
    m = iround(H / (2 * TH))
    top = m * TH
    bot = H - top
    return top, bot

def split_heights_general(H: int, TH: int) -> Tuple[List[Tuple[int, Tuple[str, ...]]], str, str]:
    """
    TH=300 → 상부 1200 모듈 반복 후 잔여(newH)에 VB(1200~2400) 또는 1장(≤1200)
    TH=250 → 상부 1000 모듈 반복, 최하부 ≤1200, 잔여(newH∈(1200,2200])는 VB
    """
    heights: List[Tuple[int, Tuple[str, ...]]] = []
    note = ""
    if TH == 300:  # 300x600
        k = max(0, math.ceil((H - 2400) / 1200))
        for _ in range(k):
            heights.append((1200, tuple()))
        newH = H - 1200 * k
        branch = f"GEN 300x600: k_upper1200={k}, newH={newH}"
        if newH > 0:
            if newH <= 1200:
                heights.append((newH, tuple()))
                note = "newH<=1200 → bottom 1장"
            else:
                top, bot = vb_round(newH, 300)
                heights.append((top, ('VB',)))
                heights.append((bot, ('VB',)))
                note = f"newH in (1200,2400] → VB ROUND (top={top}, bot={bot})"
        return heights, branch, note

    elif TH == 250:  # 250x400
        k = max(0, (H - 1200) // 1000)
        for _ in range(k):
            heights.append((1000, tuple()))
        newH = H - 1000 * k
        branch = f"GEN 250x400: k_upper1000={k}, newH={newH}"
        if newH <= 1200:
            heights.append((newH, tuple()))
            note = "newH<=1200 → bottom 1장"
        else:
            if newH > 2200:
                heights.append((1000, tuple()))
                newH -= 1000
            top, bot = vb_round(newH, 250)
            heights.append((top, ('VB',)))
            heights.append((bot, ('VB',)))
            note = f"newH in (1200,2200] → VB ROUND (top={top}, bot={bot})"
        return heights, branch, note

    else:
        raise RuleError("UNSUPPORTED_TILE_HEIGHT")

def split_heights_vstrip(H: int, TH: int) -> Tuple[List[Tuple[int, Tuple[str, ...]]], str]:
    """세로판(VSTRIP) 열의 세로 분해 (타일별 1~4판 규칙 + 아래 2판에 VB)"""
    if TH == 300:
        if H <= 2400:
            return [(H, tuple())], "VSTRIP 300x600: H<=2400 → 1판"
        if H <= 4800:
            top, bot = vb_round(H, 300)
            return [(top, ('VB',)), (bot, ('VB',))], f"VSTRIP 300x600: 2판 VB (top={top}, bot={bot})"
        if H <= 7200:
            rem = H - 1200
            top2, bot2 = vb_round(rem, 300)
            return [(1200, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 300x600: 3판 (1200 + VB on {rem} → {top2},{bot2})"
        rem = H - 2400
        top2, bot2 = vb_round(rem, 300)
        return [(1200, tuple()), (1200, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 300x600: 4판 (1200×2 + VB on {rem} → {top2},{bot2})"

    elif TH == 250:
        if H <= 2200:
            return [(H, tuple())], "VSTRIP 250x400: H<=2200 → 1판"
        if H <= 4200:
            top, bot = vb_round(H, 250)
            return [(top, ('VB',)), (bot, ('VB',))], f"VSTRIP 250x400: 2판 VB (top={top}, bot={bot})"
        if H <= 6200:
            rem = H - 1000
            top2, bot2 = vb_round(rem, 250)
            return [(1000, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 250x400: 3판 (1000 + VB on {rem} → {top2},{bot2})"
        rem = H - 2000
        top2, bot2 = vb_round(rem, 250)
        return [(1000, tuple()), (1000, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 250x400: 4판 (1000×2 + VB on {rem} → {top2},{bot2})"

    else:
        raise RuleError("UNSUPPORTED_TILE_HEIGHT")

def layout_report(W: int, H: int, TH: int, TW: int) -> Dict[str, Any]:
    """
    입력: W,H,TH,TW
    출력: {
      inputs, constraints, horiz_branch,
      columns: [{col, col_w, col_tags, col_note, vertical_rule, vertical_note}],
      panels:  [{col,row,panel_w,panel_h,col_tags,row_tags,col_note,row_note}],
      counts:  {n_cols, n_cols_vstrip, n_cols_hb, n_panels}
    }
    """
    if W <= MIN_EDGE or H <= MIN_EDGE:
        raise RuleError("W_AND_H_MUST_BE_GREATER_THAN_80")
    if (TH, TW) == (300, 600):
        if W > 9600 or H > 9600:
            raise RuleError("OUT_OF_RANGE_300x600_MAX_9600x9600")
    elif (TH, TW) == (250, 400):
        if W > 9600 or H > 8200:
            raise RuleError("OUT_OF_RANGE_250x400_MAX_9600x8200")
    else:
        raise RuleError("UNSUPPORTED_TILE_SIZE")

    cols, horiz_branch = split_columns_verbose(W, TW)

    panels: List[PanelCell] = []
    col_meta: List[Dict[str, Any]] = []
    for ci, c in enumerate(cols, start=1):
        cw, ctags, cnote = int(c["width"]), tuple(c["tags"]), c["col_note"]
        if 'VSTRIP' in ctags:
            rows, vbranch = split_heights_vstrip(H, TH)
            vbranch_general, vnote_general = "", ""
        else:
            rows, vbranch_general, vnote_general = split_heights_general(H, TH)
            vbranch = ""
        col_meta.append({
            "col": ci, "col_w": cw, "col_tags": ",".join(ctags) if ctags else "",
            "col_note": cnote,
            "vertical_rule": vbranch or vbranch_general, "vertical_note": vnote_general
        })
        for rj, (rh, rtags) in enumerate(rows, start=1):
            panels.append(PanelCell(
                col=ci, row=rj, w=cw, h=int(rh),
                col_tags=ctags, row_tags=rtags,
                col_note=cnote, row_note=(vbranch or vnote_general)
            ))

    ensure_producible_new(panels)

    n_cols = len(cols)
    n_vstrip = sum(1 for c in cols if 'VSTRIP' in c["tags"])
    n_hbcols = sum(1 for c in cols if 'HB' in c["tags"])
    out = {
        "inputs": {"W": W, "H": H, "TH": TH, "TW": TW},
        "constraints": {
            "min_edge_mm": MIN_EDGE, "range_ok": True,
            "tile_limit": "300x600: 9600×9600 / 250x400: 9600×8200"
        },
        "horiz_branch": horiz_branch,
        "columns": [{
            "col": cm["col"], "col_w": cm["col_w"],
            "col_tags": cm["col_tags"], "col_note": cm["col_note"],
            "vertical_rule": cm["vertical_rule"], "vertical_note": cm["vertical_note"],
        } for cm in col_meta],
        "panels": [p.as_dict() for p in panels],
        "counts": {"n_cols": n_cols, "n_cols_vstrip": n_vstrip, "n_cols_hb": n_hbcols, "n_panels": len(panels)}
    }
    return out

# =========================================================
# 2) 벽/벽면(Face) 모델 & 생성
# =========================================================
def validate_corner_dims(w: Dict[int, int]) -> List[str]:
    """코너형 합치 조건 검사: W1==W3+W5, W2==W4+W6, 모두>0."""
    err = []
    W1, W2, W3, W4, W5, W6 = (w[i] for i in range(1, 7))
    if any(v <= 0 for v in [W1, W2, W3, W4, W5, W6]):
        err.append("코너형 모든 벽폭(W1~W6)은 0보다 커야 합니다.")
    if W1 != W3 + W5:
        err.append(f"합치 조건 위반: W1(={W1})은 W3+W5(={W3+W5}) 이어야 합니다.")
    if W2 != W4 + W6:
        err.append(f"합치 조건 위반: W2(={W2})은 W4+W6(={W4+W6}) 이어야 합니다.")
    return err

def normalize_door(W: int, s: float, d: float) -> Tuple[float, float, float, float, int]:
    """도어 시작/폭 정규화: 반환 (s, e, L, R, n_faces)"""
    if d <= 0 or d > W:
        raise ValueError("문 폭(d)이 유효하지 않습니다. 0 < d ≤ 문벽 폭(W)을 만족해야 합니다.")
    s = max(0.0, min(float(s), float(W)))
    if s == W:
        s = float(W - d)
    e = s + d
    if e > W:
        raise ValueError("문 범위(s+d)가 문벽 폭(W)을 초과합니다. 시작점 또는 문폭을 줄이세요.")
    L = s
    R = W - e
    n_faces = (1 if L > 0 else 0) + (1 if R > 0 else 0)
    return s, e, L, R, n_faces

def rect_wall_width_of(wall_id: int, BW: int, BL: int) -> int:
    """사각형: 1=상(BL), 2=우(BW), 3=하(BL), 4=좌(BW)"""
    if wall_id == 1: return BL
    if wall_id == 2: return BW
    if wall_id == 3: return BL
    if wall_id == 4: return BW
    raise ValueError("사각형 문벽 번호는 1~4 범위여야 합니다.")

def corner_wall_width_of(wall_id: int, w: Dict[int, int]) -> int:
    """코너형: 입력 W1..W6 그대로 사용"""
    if wall_id not in w:
        raise ValueError("코너형 문/젠다이 벽 번호는 1~6 범위여야 합니다.")
    return w[wall_id]

@dataclass
class FaceSpec:
    wall_id: int
    wall_label: str  # "W1".."W6"
    face_idx: int
    face_label: str  # "W1F1".."W6F3"
    x0: int; x1: int
    y0: int; y1: int
    width_mm: int
    height_mm: int
    note: str

@st.cache_data
def parse_price_file(file_data: bytes) -> Tuple[Optional[int], str]:
    """
    엑셀 파일에서 벽판 단가를 파싱합니다.
    Streamlit cache를 사용하여 반복 파싱을 방지합니다.

    Args:
        file_data: 업로드된 파일의 바이트 데이터

    Returns:
        (단가, 메시지)
    """
    try:
        xls = pd.ExcelFile(file_data)
        if "자재단가내역" not in xls.sheet_names:
            return None, "'자재단가내역' 시트를 찾지 못해 기본단가 사용"

        df_price = pd.read_excel(xls, "자재단가내역")
        wall_rows = df_price[df_price["품목"] == "벽판"]

        if wall_rows.empty:
            return None, "엑셀에 '품목=벽판' 행이 없어 기본단가 사용"

        unit_price = int(wall_rows.iloc[0]["단가"])
        return unit_price, f"엑셀에서 벽판단가 {unit_price:,}원 적용"

    except Exception as ex:
        return None, f"엑셀 읽기 오류: {ex}"

def wall_label(shape: str, wall_id: int) -> str:
    return f"W{wall_id}"

def build_faces_for_wall(
    shape: str,
    wall_id: int,
    width_mm: int,
    height_mm: int,
    door_tuple: Optional[Tuple[float, float]] = None,   # (s,e) mm
    j_enabled: bool = False,
    j_wall: Optional[int] = None,
    j_has_step: bool = False,
    j_h: int = 1000,
    j_depth: int = 0,
    j_lower_segments: Optional[List[int]] = None,
) -> List[FaceSpec]:
    """
    한 '벽'을 문/젠다이 설정에 따라 여러 FaceSpec으로 분해한다.
    """
    wl = wall_label(shape, wall_id)
    faces: List[FaceSpec] = []

    # 0) 도어 분할
    if door_tuple is not None:
        s_mm = int(round(door_tuple[0]))
        e_mm = int(round(door_tuple[1]))
        L = max(0, s_mm)
        R = max(0, width_mm - e_mm)
        fi = 1
        if L > 0:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                0, L, 0, height_mm,
                L, height_mm, "door-left"
            ))
            fi += 1
        if R > 0:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                e_mm, e_mm + R, 0, height_mm,
                R, height_mm, "door-right"
            ))
        return faces

    # 2) 젠다이
    if j_enabled and (j_wall is not None) and (int(j_wall) == int(wall_id)) and (j_h > 0):
        fi = 1
        band_h = min(int(j_h), int(height_mm))
        if j_has_step:
            segments = [int(v) for v in (j_lower_segments or []) if int(v) > 0]
            acc = 0
            for seg_w in segments:
                seg_w = min(seg_w, int(width_mm) - acc)
                if seg_w <= 0:
                    continue
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    acc, acc + seg_w, 0, band_h,
                    seg_w, band_h, "jendai-lower"
                ))
                acc += seg_w
                fi += 1
            upper_h = max(0, int(height_mm) - band_h)
            if upper_h > 0:
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    0, int(width_mm), band_h, band_h + upper_h,
                    int(width_mm), upper_h, "jendai-upper"
                ))
        else:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                0, int(width_mm), 0, band_h,
                int(width_mm), band_h, "jendai-lower"
            ))
            fi += 1
            upper_h = max(0, int(height_mm) - band_h)
            if upper_h > 0:
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    0, int(width_mm), band_h, band_h + upper_h,
                    int(width_mm), upper_h, "jendai-upper"
                ))
        return faces

    # 3) 기본 면
    faces.append(FaceSpec(
        wall_id, wl, 1, f"{wl}F1",
        0, int(width_mm), 0, int(height_mm),
        int(width_mm), int(height_mm), "single"
    ))
    return faces

# =========================================================
# 3) 도면 렌더링
# =========================================================
def draw_rect_preview(
    BL: int, BW: int,
    has_split: bool, X: Optional[int],
    door_info: Optional[Tuple[int, float, float, int]] = None,
) -> Image.Image:
    """사각형 평면도. 라벨: W1~W4 (사각형을 조금 줄이고, 폰트는 키움)"""
    # 가로/세로 뒤집기 방지
    if BW > BL:
        BL, BW = BW, BL

    CANVAS_W = 760
    MARGIN = 60  # 기존 20 → 60 : 사각형을 줄이고 라벨 공간 확보

    # 폰트 크게 (가능하면 DejaVuSans, 없으면 기본)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("DejaVuSans.ttf", 18)  # 폰트 크기 ↑
    except Exception:
        font = None

    # 라벨 높이/여백
    if font:
        try:
            bbox = font.getbbox("W1")
            label_h = bbox[3] - bbox[1]
        except Exception:
            label_h = font.getsize("W1")[1]
    else:
        label_h = 14
    LABEL_MARGIN = 10

    # 스케일 계산 (여유 공간 남기고 사각형 축소)
    sx = (CANVAS_W - 2 * MARGIN) / max(1.0, float(BL))
    sy = sx
    rect_h_px = BW * sy
    CANVAS_H = int(rect_h_px + 2 * MARGIN + label_h)  # 아래쪽에 라벨 공간 추가

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    # 사각형 위치
    x0 = MARGIN
    y0 = MARGIN
    x1 = x0 + int(BL * sx)
    y1 = y0 + int(BW * sy)

    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    # 세면/샤워 경계선
    if has_split and X is not None:
        gx = x0 + int(X * sx)
        drw.line([gx, y0, gx, y1], fill="blue", width=3)

    # 문(도어)
    if door_info:
        wall_id, s, e, W_wall = door_info
        if wall_id == 1:
            xs = x0 + int(s * sx)
            xe = x0 + int(e * sx)
            y = y1
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 3:
            xs = x0 + int(s * sx)
            xe = x0 + int(e * sx)
            y = y0
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 2:
            ys = y0 + int(s * sy)
            ye = y0 + int(e * sy)
            x = x1
            drw.line([x, ys, x, ye], fill="red", width=5)
        elif wall_id == 4:
            ys = y0 + int(s * sy)
            ye = y0 + int(e * sy)
            x = x0
            drw.line([x, ys, x, ye], fill="red", width=5)

    # 가운데 정렬 텍스트 유틸
    def draw_centered(text: str, cx: float, cy: float):
        if font:
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = font.getsize(text)
            drw.text((cx - tw / 2, cy - th / 2), text, font=font, fill="black")
        else:
            drw.text((cx - 12, cy - 7), text, fill="black")

    # 라벨 배치 (W1~W4)
    # 아래(W1)
    draw_centered("W1", (x0 + x1) / 2, y1 + LABEL_MARGIN + label_h / 2)
    # 위(W3)
    draw_centered("W3", (x0 + x1) / 2, y0 - LABEL_MARGIN - label_h / 2)
    # 오른쪽(W2)
    draw_centered("W2", x1 + LABEL_MARGIN + label_h / 2, (y0 + y1) / 2)
    # 왼쪽(W4)
    draw_centered("W4", x0 - LABEL_MARGIN - label_h / 2, (y0 + y1) / 2)

    return img

def draw_corner_preview(
    W: dict,
    has_split: bool,
    canvas_w: int = 760,
    margin: int = 20,
) -> Image.Image:
    """
    코너형 평면도. 라벨: W1~W6
    - W1 = W3 + W5 (가로 전체)
    - W2 = W4 + W6 (세로 전체)
    - W3: 세면부 길이
    - W5: 샤워부 길이
    - W4: 오목부 폭(위쪽 빈 영역 높이)
    - W6: 샤워부 폭(아래쪽 샤워 영역 높이)
    """
    W1, W2, W3, W4, W5, W6 = (int(W[i]) for i in range(1, 7))

    CANVAS_W = int(canvas_w)
    MARGIN   = int(margin)

    # 왼쪽에 라벨 공간 확보용 오프셋
    EXTRA_X = 96

    # 가로길이 W1을 기준으로 스케일 결정 (세로는 비율 유지)
    sx = (CANVAS_W - 2 * MARGIN) / max(1.0, float(W1))
    sy = sx
    CANVAS_H = int(W2 * sy + 2 * MARGIN)

    # 실제 이미지 폭은 오른쪽 여유(EXTRA_X)를 더 줌
    IMG_W = CANVAS_W + EXTRA_X
    img = Image.new("RGB", (IMG_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    # 좌표 변환 (0,0 이 욕실 왼쪽 위 모서리라고 가정)
    x0 = MARGIN + EXTRA_X
    y0 = MARGIN

    def X(mm: float) -> int:
        return int(round(x0 + mm * sx))

    def Y(mm: float) -> int:
        return int(round(y0 + mm * sy))

    # 1) 외곽(전체 직사각형: 0~W1, 0~W2)
    drw.rectangle([X(0), Y(0), X(W1), Y(W2)], outline="black", width=3)

    # 2) 오목부(오른쪽 상단 빈 영역)
    #    가로: W3~W1   (폭 = W5)
    #    세로: 0~W4    (높이 = W4)
    notch_x0, notch_x1 = W1 - W5, W1
    notch_y0, notch_y1 = 0, W4

    # 오목부는 '천장/벽 없음' 영역이라 흰색으로 지우고 경계선 다시 그림
    drw.rectangle(
        [X(notch_x0), Y(notch_y0), X(notch_x1), Y(notch_y1)],
        fill="white",
        outline="white",
    )
    drw.line([X(notch_x0), Y(0),          X(notch_x0), Y(notch_y1)], fill="black", width=3)
    drw.line([X(notch_x0), Y(notch_y1),   X(W1),      Y(notch_y1)], fill="black", width=3)

    # 3) 샤워부(오목부 바로 아래, 오른쪽 하단)
    #    가로: W3~W1           (폭 = W5)
    #    세로: W4~W2           (높이 = W6,  W2 = W4 + W6)
    shower_x0, shower_x1 = notch_x0, W1
    shower_y0, shower_y1 = notch_y1, W2   # = W4, W2

    drw.rectangle(
        [X(shower_x0), Y(shower_y0), X(shower_x1), Y(shower_y1)],
        outline="black",
        fill="#eeeeee",
        width=1,
    )

    # 샤워부 라벨 (대략 중앙)
    cx = (shower_x0 + shower_x1) / 2.0
    cy = (shower_y0 + shower_y1) / 2.0
    drw.text((X(cx) - 18, Y(cy) - 7), "샤워부", fill="black")

    # 4) 세면/샤워 경계선 (W3 위치)
    if has_split:
        drw.line([X(W3), Y(0), X(W3), Y(W2)], fill="blue", width=3)

    # 5) 라벨 W1~W6 위치
    off = 14

    # W1: 바닥(가로 전체)
    drw.text((X(W1 / 2.0),           Y(W2) + off), "W1", fill="black")
    # W2: 왼쪽 세로 벽 전체
    drw.text((X(0) - off,            Y(W2 / 2.0)), "W2", fill="black")
    # W3: 상단 왼쪽(세면부 길이)
    drw.text((X(W3 / 2.0),           Y(0) - off),  "W3", fill="black")
    # W4: 오목부 세로폭 (오목부 왼쪽 라인 중간)
    drw.text((X(notch_x0) - off,     Y(notch_y1 / 2.0)), "W4", fill="black")
    # W5: 오목부/샤워 가로폭 (오목부/샤워 경계 아래)
    drw.text((X(W1 - W5 / 2.0),      Y(notch_y1) + off), "W5", fill="black")
    # W6: 우측 세로 벽 중 샤워부 쪽이 강조되도록 약간 아래쪽
    drw.text((X(W1) + off,           Y(W2 / 2.0) + 30),  "W6", fill="black")

    return img

def draw_wall_elevation_with_faces(
    wall_label_str: str,
    width_mm: int,
    height_mm: int,
    faces: List[FaceSpec],
    target_h_px: int = 280,
    margin: int = 16,
    overlays: Optional[List[Tuple[int,int,int,int]]] = None,
    scale: Optional[float] = None,
) -> Image.Image:
    """
    벽 정면도를 그림.
    scale이 주어지면 공통 스케일 사용 (모든 벽의 높이 기준 일관성 유지),
    아니면 개별 height_mm 기준으로 스케일 계산.
    """
    usable_h = target_h_px - 2 * margin
    s = scale if scale is not None else usable_h / max(1.0, float(height_mm))
    W = int(round(width_mm  * s))
    H = int(round(height_mm * s))
    CANVAS_W = int(W + 2 * margin)
    CANVAS_H = int(target_h_px + 28)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = margin, margin + 20
    x1, y1 = x0 + W, y0 + H

    drw.text((margin, 4), f"{wall_label_str} : {width_mm}×{height_mm} mm", fill="black")
    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    if overlays:
        for (ox0, ox1, oy0, oy1) in overlays:
            fx0 = x0 + int(round(ox0 * s))
            fx1 = x0 + int(round(ox1 * s))
            fy0 = y1 - int(round(oy0 * s))
            fy1 = y1 - int(round(oy1 * s))
            drw.rectangle([fx0, fy1, fx1, fy0], outline="black", fill="black", width=2)

    for f in faces:
        fx0 = x0 + int(round(f.x0 * s))
        fx1 = x0 + int(round(f.x1 * s))
        fy0 = y1 - int(round(f.y0 * s))
        fy1 = y1 - int(round(f.y1 * s))
        drw.rectangle([fx0, fy1, fx1, fy0], outline="#666666", width=2)
        cx = (fx0 + fx1) // 2
        cy = (fy0 + fy1) // 2
        drw.text((cx - 14, cy - 7), f.face_label, fill="black")
    return img

# ### NEW: JENDAI SIDE UTIL
def compute_jendai_side_panels(shape: str, j_enabled: bool, j_has_step: bool,
                               j_depth: int, j_h: int):
    """단차가 있는 젠다이의 '옆벽판' 자동 생성.
       - 사각형: 2장, 코너형: 1장
       - 치수: (폭=젠다이 깊이, 높이=젠다이 높이)
    """
    if not (j_enabled and j_has_step):
        return []
    cnt = 2 if shape == "사각형" else 1
    return [{
        "벽": "젠다이옆벽",
        "벽면": f"JEND_SIDE_{i+1}",
        "타일": "",
        "가로분해": "SIDE-PANEL",
        "세로규칙": "SIDE-PANEL",
        "열": 1, "행": 1,
        "panel_w": int(j_depth), "panel_h": int(j_h),
        "col_tags": "", "row_tags": "",
        "face_w": int(j_depth), "face_h": int(j_h),
    } for i in range(cnt)]

# =========================================================
# 4) 통합 파이프라인
# =========================================================
def collect_all_faces(
    shape: str,
    widths: Dict[int,int],
    H_eff: int,
    door_wall: Optional[int],
    door_s: Optional[float],
    door_e: Optional[float],
    j_enabled: bool,
    j_wall: Optional[int],
    j_has_step: bool,
    j_h: int,
    j_depth: int,
    j_lower_segments_map: Dict[int, List[int]],
) -> List[FaceSpec]:
    all_faces: List[FaceSpec] = []
    for wid, Wk in widths.items():
        door_tuple = None
        if (door_wall is not None) and (int(door_wall)==wid) and (door_s is not None) and (door_e is not None):
            door_tuple = (float(door_s), float(door_e))
        faces = build_faces_for_wall(
            shape=shape,
            wall_id=int(wid),
            width_mm=int(Wk),
            height_mm=int(H_eff),
            door_tuple=door_tuple,
            j_enabled=j_enabled,
            j_wall=(int(j_wall) if j_wall is not None else None),
            j_has_step=j_has_step,
            j_h=int(j_h),
            j_depth=int(j_depth),
            j_lower_segments=j_lower_segments_map.get(int(wid), None),
        )
        all_faces.extend(faces)
    return all_faces

def panels_for_faces_new_engine(faces: List[FaceSpec], TH: int, TW: int):
    rows, errs = [], []

    for f in faces:
        if int(f.width_mm) <= 0 or int(f.height_mm) <= 0:
            errs.append({
                "벽": f.wall_label, "벽면": f.face_label,
                "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                "타일": f"{TH}×{TW}", "error": "INVALID_FACE_SIZE", "분할사유": getattr(f, "note", "")
            })
            continue

        try:
            rpt = layout_report(int(f.width_mm), int(f.height_mm), TH, TW)
            horiz = rpt.get("horiz_branch", "")
            for p in rpt.get("panels", []):
                rows.append({
                    "벽": f.wall_label,
                    "벽면": f.face_label,
                    "타일": f"{TH}×{TW}",
                    "가로분해": horiz,
                    "세로규칙": p.get("row_note","") or "",
                    "열": p["col"], "행": p["row"],
                    "panel_w": int(p["panel_w"]), "panel_h": int(p["panel_h"]),
                    "col_tags": p.get("col_tags",""), "row_tags": p.get("row_tags",""),
                    "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                })
        except Exception as ex:
            errs.append({
                "벽": f.wall_label, "벽면": f.face_label,
                "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                "타일": f"{TH}×{TW}", "error": str(ex), "분할사유": getattr(f, "note", "")
            })

    return rows, errs

# =========================================================
# 5) UI
# =========================================================
st.title("벽판 계산 프로그램 (통합)")

# 바닥판 완료 확인
floor_done = st.session_state.get(FLOOR_DONE_KEY, False)
if not floor_done:
    st.warning("⚠️ 벽판 계산을 진행하려면 먼저 **바닥판 계산**을 완료해야 합니다.")
    st.info("""
    좌측 사이드바에서 **바닥판 계산** 페이지로 이동하여 계산을 먼저 진행하세요.
    """)
    st.stop()

# ★ 기본 단가 전역값(세션) 준비
if "wall_unit_price" not in st.session_state:
    st.session_state["wall_unit_price"] = 30000  # 기본 3만원
if "last_price_msg" not in st.session_state:
    st.session_state["last_price_msg"] = "기본단가 30,000원 사용"

# 바닥판에서 업로드한 엑셀 파일 가져오기
excel_file = st.session_state.get(SHARED_EXCEL_KEY)
excel_filename = st.session_state.get(SHARED_EXCEL_NAME_KEY, "알 수 없음")

# 엑셀 파일이 있으면 단가 읽기 (캐시된 파싱 사용)
if excel_file is not None:
    excel_file.seek(0)  # 파일 포인터를 처음으로 리셋
    file_bytes = excel_file.read()
    unit_price, msg = parse_price_file(file_bytes)
    if unit_price is not None:
        st.session_state["wall_unit_price"] = unit_price
    st.session_state["last_price_msg"] = msg

# 카탈로그 확인 UI
with st.expander("📋 업로드된 엑셀 정보 확인", expanded=False):
    st.write(f"**파일명**: {excel_filename}")
    st.markdown(f"**현재 벽판 단가**: {st.session_state['wall_unit_price']:,} 원")
    st.caption(st.session_state["last_price_msg"])

with st.sidebar:
    st.header("기본 입력")

    shape = st.radio("욕실형태", ["사각형", "코너형"], horizontal=True)
    split_kind = st.radio("세면/샤워 구분", ["구분 없음", "구분 있음"], horizontal=True)
    H = st.number_input("벽 높이 H (mm)", min_value=300, value=2200, step=50)
    floor_type = st.radio("바닥판 유형", ["PVE", "그외(GRP/FRP)"], horizontal=True)
    tile = st.selectbox("벽타일 규격", ["300×600", "250×400"])
    H_eff = effective_height(H, floor_type)

    st.divider()
    st.subheader("문(도어) 설정")
    door_wall = st.number_input("문벽 번호", min_value=1, max_value=(4 if shape=="사각형" else 6), value=1, step=1)
    door_s = st.number_input("문 시작점 s (mm)", min_value=0.0, value=0.0, step=10.0)
    door_d = st.number_input("문 폭 d (mm)", min_value=0.0, value=800.0, step=10.0)

    st.divider()
    st.subheader("젠다이")
    j_enabled = st.checkbox("젠다이 있음")
    j_wall = None
    j_has_step = False
    j_h = 1000
    j_depth = 0
    j_lower_segments_map: Dict[int, List[int]] = {}

    if j_enabled:
        j_wall = st.number_input("젠다이 벽 번호", min_value=1, max_value=(4 if shape=="사각형" else 6), value=1, step=1)
        j_h = st.number_input("젠다이 높이 (mm)", min_value=50, value=1000, step=10)
        j_depth = st.number_input("젠다이 깊이 (mm)", min_value=0, value=300, step=10)

        j_has_step = st.radio("젠다이 단차", ["없음", "있음"], horizontal=True) == "있음"
        if j_has_step:
            if shape == "사각형":
                st.markdown("하부 분할(사각형): 왼쪽→오른쪽 순서 ①②③")
                w1 = st.number_input("하부 ① 폭 (mm)", min_value=0, value=600, step=10)
                w2 = st.number_input("하부 ② 폭 (mm)", min_value=0, value=600, step=10)
                w3 = st.number_input("하부 ③ 폭 (mm)", min_value=0, value=600, step=10)
                j_lower_segments_map[int(j_wall)] = [int(w1), int(w2), int(w3)]
            else:
                st.markdown("하부 분할(코너형): 왼쪽→오른쪽 순서 ①②")
                w1 = st.number_input("하부 ① 폭 (mm)", min_value=0, value=600, step=10, key="corner_step1")
                w2 = st.number_input("하부 ② 폭 (mm)", min_value=0, value=600, step=10, key="corner_step2")
                j_lower_segments_map[int(j_wall)] = [int(w1), int(w2)]

    st.divider()
    st.subheader("비용 비율(%)")
    rp = st.number_input("생산관리비율 rₚ (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    rs = st.number_input("영업관리비율 rₛ (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

    st.divider()
    calc = st.button("계산 & 미리보기", type="primary")

errors: List[str] = []
preview_img: Optional[Image.Image] = None

# 이 변수들은 두 branch에서 공통으로 쓰려고 미리 선언
final_cost_json = None
wall_unit_price = st.session_state["wall_unit_price"]

if shape == "사각형":
    st.subheader("사각형 입력")
    colA, colB = st.columns(2)
    with colA:
        BL = st.number_input("욕실 길이 BL (mm)", min_value=500, value=2000, step=50)
    with colB:
        BW = st.number_input("욕실 폭 BW (mm)", min_value=500, value=1600, step=50)

    X = None
    if split_kind == "구분 있음":
        X = st.slider("세면/샤워 경계 위치 X (mm)", min_value=100, max_value=int(BL), step=50, value=min(800, int(BL)))

    door_W = rect_wall_width_of(int(door_wall), int(BW), int(BL))

    # 젠다이 단차 검증
    if j_enabled and j_has_step and (j_wall is not None):
        target_w = rect_wall_width_of(int(j_wall), int(BW), int(BL))
        segs = [int(x) for x in (j_lower_segments_map.get(int(j_wall), []) or [])]
        need = 3
        if len(segs) < need:
            errors.append(f"사각형 단차: 하부 분할 폭은 {need}개가 필요합니다.")
        elif sum(segs) != target_w:
            errors.append(f"하부 분할 폭 합({sum(segs)}) ≠ 해당 벽폭({target_w})")

    if j_enabled and (int(door_wall) == int(j_wall or -999)):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
            door_draw_info = (int(door_wall), s, e, int(door_W))
        except Exception as ex:
            errors.append(str(ex))
            door_draw_info = None

        if errors:
            for msg in errors: st.error(msg)
        else:
            preview_img = draw_rect_preview(
                BL=int(BL), BW=int(BW),
                has_split=(split_kind=="구분 있음"),
                X=(int(X) if X is not None else None),
                door_info=door_draw_info
            )
            st.image(preview_img, caption="사각형 도면(평면) 미리보기", width=max(160, int(preview_img.width/2)), use_container_width=False)

            widths = {1:int(BL), 2:int(BW), 3:int(BL), 4:int(BW)}
            st.subheader("벽면(정면도) / 라벨: WnF#")

            cols = st.columns(2)
            TH, TW = parse_tile(tile)

            # 공통 스케일 계산 (모든 벽의 높이 기준)
            TARGET_H = 280
            MARGIN = 16
            usable_h = TARGET_H - 2 * MARGIN
            global_scale = usable_h / float(H_eff)

            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1,2,3,4]):
                Wk = widths[wid]
                door_tuple = (float(s), float(e)) if (door_draw_info and int(door_wall)==wid) else None
                faces = collect_all_faces(
                    shape="사각형",
                    widths={wid: Wk},
                    H_eff=int(H_eff),
                    door_wall=(wid if door_tuple else None),
                    door_s=(door_tuple[0] if door_tuple else None),
                    door_e=(door_tuple[1] if door_tuple else None),
                    j_enabled=j_enabled,
                    j_wall=(int(j_wall) if j_enabled else None),
                    j_has_step=j_has_step,
                    j_h=int(j_h),
                    j_depth=int(j_depth),
                    j_lower_segments_map=j_lower_segments_map,
                )
                faces = [f for f in faces if f.wall_id == wid]
                all_faces.extend(faces)

                img = draw_wall_elevation_with_faces(
                    wall_label("사각형", wid), Wk, int(H_eff), faces,
                    target_h_px=TARGET_H,
                    margin=MARGIN,
                    scale=global_scale
                    )

                with cols[i%2]:
                    is_jendai_wall = (j_enabled and j_has_step and (j_wall is not None) and (int(j_wall) == int(wid)))
                    extra = 2 if is_jendai_wall else 0
                    caption = f"{wall_label('사각형', wid)} (벽면 {len(faces) + extra})개"
                    st.image(img, caption=caption, use_container_width=False)

            # 새 엔진으로 패널 산출
            st.subheader("벽면별 벽판 산출 (New Engine)")
            rows, errs = panels_for_faces_new_engine(all_faces, TH, TW)
            if j_enabled and j_has_step and int(j_depth) > 0 and int(j_h) > 0:
                side_rows = compute_jendai_side_panels("사각형", j_enabled, j_has_step, int(j_depth), int(j_h))
                TH, TW = parse_tile(tile)
                for r in side_rows:
                    r["타일"] = f"{TH}×{TW}"
                rows.extend(side_rows)

            if rows:
                df = (pd.DataFrame(rows)
                      .rename(columns={
                          "face_w": "벽면폭", "face_h": "벽면높이",
                          "panel_w": "벽판폭", "panel_h": "벽판높이",
                          "가로분해": "가로분해(분기)", "세로규칙": "세로규칙(노트)"
                      }))
                show_cols = ["벽","벽면","타일","가로분해(분기)","세로규칙(노트)","열","행","벽판폭","벽판높이","벽면폭","벽면높이","col_tags","row_tags"]
                df = df[[c for c in show_cols if c in df.columns]]
                st.dataframe(df, use_container_width=True)

                # 비용계산
                panel_count = len(df)
                subtotal = panel_count * wall_unit_price  # 소계
                r_p = rp / 100.0
                r_s = rs / 100.0

                prod_included = subtotal / (1 - r_p) if r_p < 1 else subtotal
                prod_cost = prod_included - subtotal
                sales_included = prod_included / (1 - r_s) if r_s < 1 else prod_included
                sales_cost = sales_included - prod_included

                st.markdown("#### 비용 집계")
                st.write(f"- 벽판 개수: **{panel_count} 장**")
                st.write(f"- 단가(1장): **{wall_unit_price:,} 원**")
                st.write(f"- 소계: **{subtotal:,} 원**")
                st.write(f"- 생산관리비({rp:.1f}%): **{prod_cost:,.0f} 원** → 생산관리비포함: **{prod_included:,.0f} 원**")
                st.write(f"- 영업관리비({rs:.1f}%): **{sales_cost:,.0f} 원** → 최종(영업관리비포함): **{sales_included:,.0f} 원**")

                # JSON으로 만들기
                final_cost_json = {
                    "panel_count": int(panel_count),
                    "unit_price": int(wall_unit_price),
                    "subtotal": int(round(subtotal)),
                    "r_p": float(r_p),
                    "production_overhead": int(round(prod_cost)),
                    "price_with_production": int(round(prod_included)),
                    "r_s": float(r_s),
                    "sales_overhead": int(round(sales_cost)),
                    "final_price": int(round(sales_included)),
                }

                # 세션 상태에 결과 저장
                st.session_state[WALL_RESULT_KEY] = {
                    "section": "wall",
                    "inputs": {
                        "shape": shape,
                        "H": H,
                        "floor_type": floor_type,
                        "tile": tile,
                        "BL": BL,
                        "BW": BW,
                        "door_wall": door_wall,
                        "door_s": door_s,
                        "door_d": door_d,
                        "j_enabled": j_enabled,
                        "j_wall": j_wall,
                        "j_h": j_h,
                        "j_depth": j_depth,
                        "j_has_step": j_has_step,
                        "rp": rp,
                        "rs": rs,
                    },
                    "result": final_cost_json,
                    "panels": df.to_dict("records"),
                }
                st.session_state[WALL_DONE_KEY] = True

                # JSON 파일로 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"wall_{timestamp}.json"
                json_path = os.path.join(EXPORT_DIR, json_filename)
                _save_json(json_path, st.session_state[WALL_RESULT_KEY])

                json_str = json.dumps(final_cost_json, ensure_ascii=False, indent=2)
                st.download_button("📥 결과 JSON 다운로드", data=json_str, file_name="wall_panel_cost.json", mime="application/json")

                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (df.groupby(["벽판폭","벽판높이"], as_index=False)
                           .size().rename(columns={"size":"qty"}))
                order["치수"] = order["벽판폭"].astype(int).astype(str) + "×" + order["벽판높이"].astype(int).astype(str)
                order = order[["치수","qty","벽판폭","벽판높이"]]
                st.dataframe(order, use_container_width=True)
                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

                st.success("✅ 계산 완료")

                # 다음 단계 안내
                st.info("""
                **다음 단계**: 천장판 계산을 진행하세요.

                좌측 사이드바에서 **천장판 계산** 페이지로 이동하여 계산을 진행할 수 있습니다.
                """)

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                df_err = (pd.DataFrame(errs).rename(columns={"face_w":"벽면폭","face_h":"벽면높이"}))
                st.dataframe(df_err, use_container_width=True)

else:
    # 코너형
    st.subheader("코너형 입력 (W1~W6)")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**가로(바닥) 방향**")
        W3 = st.number_input("W3 (mm)", min_value=100, value=800, step=50, key="corner_w3")
        W5 = st.number_input("W5 (mm)", min_value=100, value=1200, step=50, key="corner_w5")
        W1 = W3 + W5
        st.text_input("W1 = W3 + W5", value=str(W1), disabled=True)
    with cB:
        st.markdown("**세로(좌우) 방향**")
        W4 = st.number_input("W4 (mm)", min_value=100, value=600, step=50, key="corner_w4")
        W6 = st.number_input("W6 (mm)", min_value=100, value=1000, step=50, key="corner_w6")
        W2 = W4 + W6
        st.text_input("W2 = W4 + W6", value=str(W2), disabled=True)

    W = {1:int(W1), 2:int(W2), 3:int(W3), 4:int(W4), 5:int(W5), 6:int(W6)}
    door_W = corner_wall_width_of(int(door_wall), W)

    if j_enabled and j_has_step and (j_wall is not None):
        target_w = corner_wall_width_of(int(j_wall), W)
        segs = [int(x) for x in (j_lower_segments_map.get(int(j_wall), []) or [])]
        need = 2
        if len(segs) < need:
            errors.append(f"코너형 단차: 하부 분할 폭은 {need}개가 필요합니다.")
        elif sum(segs) != target_w:
            errors.append(f"하부 분할 폭 합({sum(segs)}) ≠ 해당 벽폭({target_w})")

    if j_enabled and 'j_wall' in locals() and int(door_wall) == int(j_wall):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        errors.extend(validate_corner_dims(W))
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
        except Exception as ex:
            errors.append(str(ex))

        if errors:
            for msg in errors: st.error(msg)
        else:
            preview_img = draw_corner_preview(W=W, has_split=(split_kind=="구분 있음"), canvas_w=480, margin = 30)
            st.image(preview_img, caption="코너형 도면(평면) 미리보기", width=preview_img.width)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            widths = {i:int(W[i]) for i in range(1,7)}
            st.subheader("벽면(정면도) / 라벨: WnF#")
            cols = st.columns(3)
            TH, TW = parse_tile(tile)

            # 공통 스케일 계산 (모든 벽의 높이 기준)
            TARGET_H = 280
            MARGIN = 16
            usable_h = TARGET_H - 2 * MARGIN
            global_scale = usable_h / float(H_eff)

            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1,2,3,4,5,6]):
                Wk = widths[wid]
                door_tuple = (float(s), float(e)) if int(door_wall)==wid else None
                faces = collect_all_faces(
                    shape="코너형",
                    widths={wid: Wk},
                    H_eff=int(H_eff),
                    door_wall=(wid if door_tuple else None),
                    door_s=(door_tuple[0] if door_tuple else None),
                    door_e=(door_tuple[1] if door_tuple else None),
                    j_enabled=j_enabled,
                    j_wall=(int(j_wall) if j_enabled else None),
                    j_has_step=j_has_step,
                    j_h=int(j_h),
                    j_depth=int(j_depth),
                    j_lower_segments_map=j_lower_segments_map,
                )
                faces = [f for f in faces if f.wall_id == wid]
                all_faces.extend(faces)

                img = draw_wall_elevation_with_faces(
                    wall_label("코너형", wid), Wk, int(H_eff), faces,
                    target_h_px=TARGET_H,
                    margin=MARGIN,
                    scale=global_scale
                    )

                with cols[i % 3]:
                    is_jendai_wall = (j_enabled and j_has_step and (j_wall is not None) and (int(j_wall) == int(wid)))
                    extra = 1 if is_jendai_wall else 0
                    caption = f"{wall_label('코너형', wid)} (벽면 {len(faces) + extra})개"
                    st.image(img, caption=caption, use_container_width=False)

            st.subheader("벽면별 벽판 산출 (New Engine)")
            rows, errs = panels_for_faces_new_engine(all_faces, TH, TW)
            if j_enabled and j_has_step and int(j_depth) > 0 and int(j_h) > 0:
                side_rows = compute_jendai_side_panels("코너형", j_enabled, j_has_step, int(j_depth), int(j_h))
                TH, TW = parse_tile(tile)
                for r in side_rows:
                    r["타일"] = f"{TH}×{TW}"
                rows.extend(side_rows)

            if rows:
                df = (pd.DataFrame(rows)
                      .rename(columns={
                          "face_w":"벽면폭","face_h":"벽면높이",
                          "panel_w":"벽판폭","panel_h":"벽판높이",
                          "가로분해":"가로분해(분기)","세로규칙":"세로규칙(노트)"
                      }))
                show_cols = ["벽","벽면","타일","가로분해(분기)","세로규칙(노트)","열","행","벽판폭","벽판높이","벽면폭","벽면높이","col_tags","row_tags"]
                df = df[[c for c in show_cols if c in df.columns]]
                st.dataframe(df, use_container_width=True)

                # 코너형도 비용계산 동일하게
                panel_count = len(df)
                subtotal = panel_count * wall_unit_price
                r_p = rp / 100.0
                r_s = rs / 100.0

                prod_included = subtotal / (1 - r_p) if r_p < 1 else subtotal
                prod_cost = prod_included - subtotal
                sales_included = prod_included / (1 - r_s) if r_s < 1 else prod_included
                sales_cost = sales_included - prod_included

                st.markdown("#### 비용 집계")
                st.write(f"- 벽판 개수: **{panel_count} 장**")
                st.write(f"- 단가(1장): **{wall_unit_price:,} 원**")
                st.write(f"- 소계: **{subtotal:,} 원**")
                st.write(f"- 생산관리비({rp:.1f}%): **{prod_cost:,.0f} 원** → 생산관리비포함: **{prod_included:,.0f} 원**")
                st.write(f"- 영업관리비({rs:.1f}%): **{sales_cost:,.0f} 원** → 최종(영업관리비포함): **{sales_included:,.0f} 원**")

                final_cost_json = {
                    "panel_count": int(panel_count),
                    "unit_price": int(wall_unit_price),
                    "subtotal": int(round(subtotal)),
                    "r_p": float(r_p),
                    "production_overhead": int(round(prod_cost)),
                    "price_with_production": int(round(prod_included)),
                    "r_s": float(r_s),
                    "sales_overhead": int(round(sales_cost)),
                    "final_price": int(round(sales_included)),
                }

                # 세션 상태에 결과 저장
                st.session_state[WALL_RESULT_KEY] = {
                    "section": "wall",
                    "inputs": {
                        "shape": shape,
                        "H": H,
                        "floor_type": floor_type,
                        "tile": tile,
                        "W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6,
                        "door_wall": door_wall,
                        "door_s": door_s,
                        "door_d": door_d,
                        "j_enabled": j_enabled,
                        "j_wall": j_wall,
                        "j_h": j_h,
                        "j_depth": j_depth,
                        "j_has_step": j_has_step,
                        "rp": rp,
                        "rs": rs,
                    },
                    "result": final_cost_json,
                    "panels": df.to_dict("records"),
                }
                st.session_state[WALL_DONE_KEY] = True

                # JSON 파일로 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"wall_{timestamp}.json"
                json_path = os.path.join(EXPORT_DIR, json_filename)
                _save_json(json_path, st.session_state[WALL_RESULT_KEY])

                json_str = json.dumps(final_cost_json, ensure_ascii=False, indent=2)
                st.download_button("📥 결과 JSON 다운로드", data=json_str, file_name="wall_panel_cost.json", mime="application/json")

                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (df.groupby(["벽판폭","벽판높이"], as_index=False)
                           .size().rename(columns={"size":"qty"}))
                order["치수"] = order["벽판폭"].astype(int).astype(str) + "×" + order["벽판높이"].astype(int).astype(str)
                order = order[["치수","qty","벽판폭","벽판높이"]]
                st.dataframe(order, use_container_width=True)
                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

                st.success("✅ 계산 완료")

                # 다음 단계 안내
                st.info("""
                **다음 단계**: 천장판 계산을 진행하세요.

                좌측 사이드바에서 **천장판 계산** 페이지로 이동하여 계산을 진행할 수 있습니다.
                """)

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                st.dataframe(pd.DataFrame(errs).rename(columns={"face_w":"벽면폭","face_h":"벽면높이"}), use_container_width=True)

st.caption("※ 새 엔진 적용 + 벽판 단가/소계/생산·영업관리비 자동계산 + JSON 내보내기까지 포함.")
