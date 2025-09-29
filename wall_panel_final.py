# wall_panel_final.py (수정됨: 순수 로직 모듈)
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PIL import Image, ImageDraw


# =========================================================
# 0) 공통 유틸
# =========================================================
def parse_tile(tile_str: str) -> Tuple[int, int]:
    # ... (생략된 기존 로직: 타일 문자열 파싱)
    a, b = tile_str.replace("x", "×").split("×")
    return int(a), int(b)


def effective_height(H: int, floor_type: str) -> int:
    """바닥판 유형이 PVE면 +50."""
    return int(H) + 50 if floor_type.upper() == "PVE" else int(H)


# =========================================================
# 1) 데이터 모델 및 엔진
# =========================================================
MIN_EDGE = 80


def iround(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass
class Panel:
    kind: str
    pos: str
    w: int
    h: int

    def label(self) -> str:
        base = self.kind
        if self.pos:
            base += f"({self.pos})"
        return f"{base} {self.w}×{self.h}"


class RuleError(Exception):
    pass


def vertical_balance_round(H_target: int, TH: int) -> Tuple[int, int, int]:
    # ... (생략된 기존 로직: 수직 분할 로직)
    return 2, H_target // 2, H_target - H_target // 2  # 임시 반환


def horizontal_balance_round(W_target: int, TW: int) -> Tuple[int, int, int]:
    # ... (생략된 기존 로직: 수평 분할 로직)
    return 2, W_target // 2, W_target - W_target // 2  # 임시 반환


def layout_300x600(W: int, H: int) -> Tuple[List[Panel], str]:
    # ... (생략된 기존 로직: 300x600 분할 로직)
    return [Panel("기본", "", 300, 600)], "300x600 규칙 (로직 생략)"


def layout_250x400(W: int, H: int) -> Tuple[List[Panel], str]:
    # ... (생략된 기존 로직: 250x400 분할 로직)
    return [Panel("기본", "", 250, 400)], "250x400 규칙 (로직 생략)"


def compute_layout(W: int, H: int, TH: int, TW: int) -> Tuple[List[Panel], str]:
    """메인 계산 함수: 벽 폭/높이와 타일 규격에 따라 패널을 분할하고 결과를 반환합니다."""
    if (TH, TW) == (300, 600):
        return layout_300x600(W, H)
    elif (TH, TW) == (250, 400):
        return layout_250x400(W, H)
    raise RuleError("UNSUPPORTED_TILE_SIZE")


# =========================================================
# 2) 도식 유틸 (PIL Image 객체를 반환하는 함수만 유지)
# =========================================================
def draw_face(
    W: int, H: int, panels: List[Panel], target_h_px: int = 280
) -> Image.Image:
    """패널 분할 결과를 도식화하는 함수."""
    # ... (생략된 기존 로직: PIL 이미지 생성 및 패널 그리기)
    scale = target_h_px / H if H else 1
    canvas_w = int(W * scale)
    canvas_h = target_h_px
    img = Image.new("RGB", (canvas_w + 50, canvas_h + 50), color="white")  # 여백 추가
    drw = ImageDraw.Draw(img)
    drw.rectangle([25, 25, canvas_w + 25, canvas_h + 25], outline="black", width=2)
    drw.text((25 + canvas_w / 2, 10), f"W:{W}mm", fill="black")
    drw.text((10, 25 + canvas_h / 2), f"H:{H}mm", fill="black")
    return img
