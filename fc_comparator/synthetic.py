"""Synthetic inspection-board renderer.

Produces photo-like boards (white board, fiducials, green connectors, dark
cables, clips with per-instance jitter and sensor noise) together with the
exact clip boxes, i.e. perfectly "marked" images. Used for the sample data, the
end-to-end tests and demos without a camera. The cable/row spacing can be
changed to imitate part numbers with different layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import FORK, FORK_LEFT, FORK_RIGHT, MISSING, ROUND, SMALL, Box

WIDTH, HEIGHT = 1280, 960
CONNECTOR_Y = (95, 185)

# Clip extents around the clip centre at scale 1 (x0, y0, x1, y1), before per-clip rotation.
_EXTENTS = {
    ROUND: (-30, -30, 30, 30),
    FORK_RIGHT: (-14, -38, 56, 38),
    FORK: (-14, -38, 56, 38),
    FORK_LEFT: (-56, -38, 14, 38),
    SMALL: (-18, -30, 18, 26),
}


@dataclass(frozen=True)
class Geometry:
    """Where cables and rows are on the board (pixels)."""

    cable_x0: float = 260
    cable_dx: float = 260
    row_y0: float = 300
    row_dy: float = 170

    def cable_x(self, cable: int) -> float:
        return self.cable_x0 + (cable - 1) * self.cable_dx

    def row_y(self, row: int) -> float:
        return self.row_y0 + (row - 1) * self.row_dy


DEFAULT_GEOMETRY = Geometry()


def cable_x(cable: int) -> float:
    return DEFAULT_GEOMETRY.cable_x(cable)


def row_y(row: int) -> float:
    return DEFAULT_GEOMETRY.row_y(row)


def render_board(columns: list[list[str]], **kwargs) -> np.ndarray:
    """Render a board image. See :func:`render_marked_board` for the arguments."""
    return render_marked_board(columns, **kwargs)[0]


def render_marked_board(
    columns: list[list[str]],
    *,
    shift: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    seed: int = 0,
    noise: float = 3.0,
    jitter: float = 3.0,
    geometry: Geometry = DEFAULT_GEOMETRY,
    extra_clips: list[tuple[str, float, float]] = (),
) -> tuple[np.ndarray, list[Box]]:
    """Render a board and return (image, clip boxes).

    ``columns[cable-1][row-1]`` is the clip label at that position (``missing``
    draws nothing). ``shift``/``angle`` move the whole board (as if it was placed
    off-centre); ``jitter`` moves each clip a little, like real assembly.
    ``extra_clips`` places additional clips (label, x, y) anywhere on the board.
    """
    rng = np.random.default_rng(seed)
    img = np.full((HEIGHT, WIDTH, 3), 246, np.uint8)
    _draw_fiducials(img)
    clips: list[tuple[str, float, float, float]] = []  # label, cx, cy, scale

    for ci, col in enumerate(columns):
        x = geometry.cable_x(ci + 1)
        _draw_connector(img, int(x), rng)
        bottom = geometry.row_y(len(col)) + 90
        cv2.line(img, (int(x), CONNECTOR_Y[1]), (int(x), int(bottom)), (45, 45, 50), 9, cv2.LINE_AA)
        for ri, label in enumerate(col):
            y = geometry.row_y(ri + 1)
            jx, jy = rng.uniform(-jitter, jitter, 2)
            scale = rng.uniform(0.94, 1.06)
            rot = rng.uniform(-4, 4)
            _draw_clip(img, label, (x + jx, y + jy), scale, rot, rng)
            if label != MISSING:
                clips.append((label, x + jx, y + jy, scale))
    for label, x, y in extra_clips:
        _draw_clip(img, label, (x, y), 1.0, 0.0, rng)
        clips.append((label, x, y, 1.0))

    # uneven lighting + sensor noise
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    gx, gy = rng.uniform(-0.06, 0.06, 2)
    light = 1.0 + gx * (xx / WIDTH - 0.5) + gy * (yy / HEIGHT - 0.5)
    out = img.astype(np.float32) * light[..., None]
    if noise > 0:
        out += rng.normal(0, noise, out.shape)
    img = np.clip(out, 0, 255).astype(np.uint8)

    m = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, 1.0)
    m[:, 2] += shift
    if shift != (0.0, 0.0) or angle:
        img = cv2.warpAffine(img, m, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderValue=(246, 246, 246))

    boxes = []
    for label, cx, cy, s in clips:
        x0, y0, x1, y1 = (v * s for v in _EXTENTS[label])
        mx, my = m @ np.array([cx, cy, 1.0])
        pad = 2.0  # small margin for the per-clip rotation
        boxes.append(Box(label, mx + x0 - pad, my + y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad))
    return img, boxes


def board_for_patterns(patterns: list[list[str]], orient_forks: str = "left") -> list[list[str]]:
    """Turn master patterns (one per cable) into drawable labels ('fork' -> fork_left/right)."""
    concrete = {"left": FORK_LEFT, "right": FORK_RIGHT}[orient_forks]
    return [[concrete if lbl == FORK else lbl for lbl in p] for p in patterns]


# ---------------------------------------------------------------------------
def _draw_fiducials(img: np.ndarray) -> None:
    rng = np.random.default_rng(12345)  # fixed: fiducials are printed on the board
    for (x, y) in [(40, 40), (WIDTH - 130, 40), (40, HEIGHT - 130), (WIDTH - 130, HEIGHT - 130)]:
        cells = rng.integers(0, 2, (6, 6))
        cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 1
        for i in range(6):
            for j in range(6):
                if cells[i, j]:
                    cv2.rectangle(img, (x + j * 15, y + i * 15), (x + j * 15 + 14, y + i * 15 + 14), (20, 20, 20), -1)
    cv2.putText(img, "FC-BOARD 01", (WIDTH // 2 - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2, cv2.LINE_AA)
    for i, x in enumerate(range(170, WIDTH - 160, 20)):  # ruler
        h = 22 if i % 5 == 0 else 12
        cv2.line(img, (x, HEIGHT - 40), (x, HEIGHT - 40 - h), (40, 40, 40), 2)


def _draw_connector(img: np.ndarray, x: int, rng: np.random.Generator) -> None:
    y0, y1 = CONNECTOR_Y
    g = int(rng.integers(140, 170))
    cv2.rectangle(img, (x - 45, y0), (x + 45, y1), (40, g, 40), -1)
    cv2.rectangle(img, (x - 45, y0), (x + 45, y1), (20, 90, 20), 2)
    for i in range(4):
        cv2.rectangle(img, (x - 35 + i * 19, y0 + 15), (x - 25 + i * 19, y0 + 32), (15, 50, 15), -1)
    cv2.line(img, (x - 30, y1 - 18), (x + 30, y1 - 18), (25, 80, 25), 3)


def _poly(points: list[tuple[float, float]], center, scale, rot) -> np.ndarray:
    a = np.deg2rad(rot)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (np.array(points, np.float32) * scale) @ r.T + np.array(center, np.float32)
    return np.round(pts * 16).astype(np.int32)  # 4 fractional bits for anti-aliased drawing


def _fill(img, points, center, scale, rot, color) -> None:
    cv2.fillPoly(img, [_poly(points, center, scale, rot)], color, cv2.LINE_AA, shift=4)


def _draw_clip(img, label, center, scale, rot, rng) -> None:
    shade = int(rng.integers(20, 45))
    dark, mid = (shade, shade, shade + 5), (shade + 60, shade + 60, shade + 65)
    cx, cy = center
    if label == ROUND:
        r = 30 * scale
        cv2.circle(img, (int(cx), int(cy)), int(r), dark, -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), int(r * 0.55), mid, 3, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), int(r * 0.18), mid, -1, cv2.LINE_AA)
    elif label in (FORK_LEFT, FORK_RIGHT, FORK):
        # U-shaped bracket: a back plate on the cable and two prongs; the open side faces
        # right for fork_right and left for fork_left.
        d = 1 if label in (FORK_RIGHT, FORK) else -1
        back = [(-14 * d, -38), (2 * d, -38), (2 * d, 38), (-14 * d, 38)]
        top = [(-14 * d, -38), (50 * d, -38), (50 * d, -22), (-14 * d, -22)]
        bot = [(-14 * d, 22), (50 * d, 22), (50 * d, 38), (-14 * d, 38)]
        tip_t = [(50 * d, -38), (56 * d, -34), (56 * d, -26), (50 * d, -22)]
        tip_b = [(50 * d, 22), (56 * d, 26), (56 * d, 34), (50 * d, 38)]
        for p in (back, top, bot, tip_t, tip_b):
            _fill(img, p, center, scale, rot, dark)
        _fill(img, [(-10 * d, -8), (-4 * d, -8), (-4 * d, 8), (-10 * d, 8)], center, scale, rot, mid)
    elif label == SMALL:
        # fir-tree arrow: stacked barbs pointing up plus a short stem
        _fill(img, [(-5, 0), (5, 0), (5, 26), (-5, 26)], center, scale, rot, dark)
        for k, w in enumerate((10, 14, 18)):
            y = -26 + k * 10
            _fill(img, [(0, y - 4), (w, y + 10), (-w, y + 10)], center, scale, rot, dark)
    elif label == MISSING:
        pass
    else:
        raise ValueError(f"Cannot draw clip {label!r}")
