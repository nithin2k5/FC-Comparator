"""Automatic board layout: which cable and row each clip belongs to.

Two jobs:

``infer_grid``
    Group clip boxes into cables (columns, left to right) and rows (top to
    bottom) purely from their positions. Used to turn a *marked good board*
    into a part's master (pattern + layout), and to inspect parts that have a
    pattern but no master image yet.

``match_layout``
    Fit a part's master layout onto the clips detected in a new image. The board
    may be shifted or slightly rotated, clips may be missing and unexpected
    clips may be present:

    1. every (expected position, detection) pair votes for a translation; the
       translation that brings the most positions within tolerance wins
       (robust to missing/extra clips - a RANSAC over translations);
    2. with >= 3 matches a similarity transform (shift + rotation + scale) is
       estimated and checked against the configured limits;
    3. positions and detections are paired by optimal assignment; pairs farther
       apart than the tolerance are rejected.

    Unmatched positions are *missing*; unmatched detections are *extras*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import LayoutConfig
from .models import Box, Layout, PlacementInfo, Taxonomy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is optional
    linear_sum_assignment = None


class LayoutError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Grid inference
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    cables: int
    rows: int
    cells: dict[tuple[int, int], list[int]]  # (cable, row) -> indices into the boxes
    centers: dict[tuple[int, int], tuple[float, float]]
    clip_size: tuple[float, float]


def clip_size(boxes: list[Box]) -> tuple[float, float]:
    if not boxes:
        return (0.0, 0.0)
    return float(np.median([b.w for b in boxes])), float(np.median([b.h for b in boxes]))


def _cluster_1d(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """Split sorted values wherever consecutive values are more than ``gap`` apart."""
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for prev, idx in zip(order, order[1:]):
        if values[idx] - values[prev] > gap:
            groups.append(np.array(current))
            current = []
        current.append(idx)
    groups.append(np.array(current))
    return groups


def infer_grid(boxes: list[Box], cables: int | None = None, rows: int | None = None) -> Grid:
    """Group boxes into a cables x rows grid.

    Raises LayoutError when there are no boxes or the number of columns/rows
    found differs from the expected ``cables``/``rows``.
    """
    if not boxes:
        raise LayoutError("no clips found")
    cw, ch = clip_size(boxes)
    centers = np.array([b.center for b in boxes], dtype=float)
    col_groups = _cluster_1d(centers[:, 0], gap=max(cw * 1.2, 20.0))
    row_groups = _cluster_1d(centers[:, 1], gap=max(ch * 0.8, 15.0))
    if cables is not None and len(col_groups) != cables:
        raise LayoutError(f"found {len(col_groups)} cable column(s), expected {cables}")
    if rows is not None and len(row_groups) != rows:
        raise LayoutError(f"found {len(row_groups)} clip row(s), expected {rows}")

    col_of = {int(i): c + 1 for c, g in enumerate(col_groups) for i in g}
    row_of = {int(i): r + 1 for r, g in enumerate(row_groups) for i in g}
    col_x = [float(centers[g, 0].mean()) for g in col_groups]
    row_y = [float(centers[g, 1].mean()) for g in row_groups]

    cells: dict[tuple[int, int], list[int]] = {}
    for i in range(len(boxes)):
        cells.setdefault((col_of[i], row_of[i]), []).append(i)
    grid_centers = {}
    for c in range(1, len(col_groups) + 1):
        for r in range(1, len(row_groups) + 1):
            idx = cells.get((c, r))
            if idx:
                mx, my = centers[idx].mean(axis=0)
                grid_centers[(c, r)] = (float(mx), float(my))
            else:
                grid_centers[(c, r)] = (col_x[c - 1], row_y[r - 1])
    return Grid(len(col_groups), len(row_groups), cells, grid_centers, (cw, ch))


def master_from_boxes(
    boxes: list[Box], image_size: tuple[int, int], cables: int | None = None, rows: int | None = None
) -> tuple[list[list[str]], Layout]:
    """Build (pattern, layout) from the clips on a known-good board.

    ``pattern[row - 1][cable - 1]`` is the clip at each position. Every position
    must hold exactly one clip.
    """
    grid = infer_grid(boxes, cables=cables, rows=rows)
    problems = []
    for c in range(1, grid.cables + 1):
        for r in range(1, grid.rows + 1):
            n = len(grid.cells.get((c, r), []))
            if n == 0:
                problems.append(f"cable {c} row {r} has no clip")
            elif n > 1:
                problems.append(f"cable {c} row {r} has {n} clips")
    if problems:
        raise LayoutError("not a complete good board: " + "; ".join(problems[:6]))
    pattern = [[boxes[grid.cells[(c, r)][0]].label for c in range(1, grid.cables + 1)]
               for r in range(1, grid.rows + 1)]
    layout = Layout(grid.cables, grid.rows, dict(grid.centers), grid.clip_size, (int(image_size[0]), int(image_size[1])))
    return pattern, layout


# ---------------------------------------------------------------------------
# Layout matching
# ---------------------------------------------------------------------------
@dataclass
class Placement:
    assignment: dict[tuple[int, int], int | None]  # position -> detection index (None = missing)
    expected: dict[tuple[int, int], tuple[float, float]]  # expected centres in the inspected image
    extras: list[int] = field(default_factory=list)  # detections not assigned to any position
    info: PlacementInfo = field(default_factory=PlacementInfo)


def _assign(P: np.ndarray, D: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Pairs (i, j) of positions P[i] and detections D[j] closer than tol, minimising total distance."""
    if len(P) == 0 or len(D) == 0:
        return []
    dist = np.linalg.norm(P[:, None, :] - D[None, :, :], axis=2)
    if linear_sum_assignment is not None:
        cost = np.where(dist <= tol, dist, 1e6)
        rows, cols = linear_sum_assignment(cost)
        return [(int(i), int(j)) for i, j in zip(rows, cols) if dist[i, j] <= tol]
    pairs, used_i, used_j = [], set(), set()  # greedy fallback
    for i, j in zip(*np.unravel_index(np.argsort(dist, axis=None), dist.shape)):
        if dist[i, j] > tol:
            break
        if i not in used_i and j not in used_j:
            pairs.append((int(i), int(j)))
            used_i.add(i)
            used_j.add(j)
    return pairs


def _best_translation(P: np.ndarray, D: np.ndarray, tol: float, max_shift: float) -> np.ndarray:
    cands = (D[None, :, :] - P[:, None, :]).reshape(-1, 2)
    cands = cands[np.linalg.norm(cands, axis=1) <= max_shift]
    cands = np.vstack([np.zeros((1, 2)), cands])
    best, best_key = cands[0], (-1, 0.0)
    for start in range(0, len(cands), 256):  # chunked to bound memory on large layouts
        chunk = cands[start:start + 256]
        Q = P[None, :, :] + chunk[:, None, :]  # K x N x 2
        d = np.linalg.norm(Q[:, :, None, :] - D[None, None, :, :], axis=3).min(axis=2)  # K x N
        inl = d <= tol
        counts = inl.sum(axis=1)
        mean_d = np.where(inl, d, 0).sum(axis=1) / np.maximum(counts, 1)
        for k in range(len(chunk)):
            key = (int(counts[k]), -float(mean_d[k]))
            if key > best_key:
                best_key, best = key, chunk[k]
    return best


def match_layout(
    detections: list[Box],
    layout: Layout,
    cfg: LayoutConfig,
    image_size: tuple[int, int] | None = None,
    expected_labels: dict[tuple[int, int], str] | None = None,
    taxonomy: Taxonomy | None = None,
) -> Placement:
    """Fit ``layout`` onto ``detections``.

    With ``expected_labels`` (+ ``taxonomy``) the geometry is refined only from
    *anchor* clips - detections whose class agrees with the master at that
    position - so wrong clips (a different part on one cable) cannot bias it.
    """
    keys = sorted(layout.positions)
    P = np.array([layout.positions[k] for k in keys], dtype=float)
    size = float(np.mean(layout.clip_size))
    if image_size and layout.image_size[0] > 0:
        s = image_size[0] / layout.image_size[0]  # different camera resolution than the master
        P, size = P * s, size * s
    # A wrong clip type can sit ~half a clip away from the master clip's centre, so the
    # tolerance is about one clip size - but never so large that neighbours could be confused.
    tol = cfg.match_tolerance * size
    if len(P) > 1:
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        tol = min(tol, 0.45 * float(d[d > 0].min()))
    info = PlacementInfo(expected=len(keys))
    if not detections:
        info.ok, info.message = False, "no clips detected"
        return Placement({k: None for k in keys}, {k: tuple(p) for k, p in zip(keys, P)}, [], info)

    D = np.array([b.center for b in detections], dtype=float)
    t = _best_translation(P, D, tol, cfg.max_shift_px)
    Q = P + t
    pairs = _assign(Q, D, tol)
    rotation, scale = 0.0, 1.0

    def anchors(prs):
        if expected_labels is None or taxonomy is None:
            return prs
        good = [(i, j) for i, j in prs if taxonomy.matches(expected_labels.get(keys[i], ""), detections[j].label)]
        return good if len(good) >= 3 else prs

    if pairs:  # robust translation from the anchor clips
        a = anchors(pairs)
        t = np.median(np.array([D[j] - P[i] for i, j in a]), axis=0)
        Q = P + t
        pairs = _assign(Q, D, tol)

    a = anchors(pairs)
    if len(a) >= 3:
        src = np.array([P[i] for i, _ in a], dtype=np.float32)
        dst = np.array([D[j] for _, j in a], dtype=np.float32)
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=max(2.0, tol / 2))
        if m is not None:
            s_fit = math.hypot(m[0, 0], m[1, 0])
            r_fit = math.degrees(math.atan2(m[1, 0], m[0, 0]))
            if abs(r_fit) <= cfg.max_rotation_deg and abs(s_fit - 1) <= 0.15:
                Q2 = P @ m[:, :2].T + m[:, 2]
                pairs2 = _assign(Q2, D, tol)
                if len(pairs2) >= len(pairs):
                    Q, pairs, rotation, scale = Q2, pairs2, r_fit, s_fit

    assignment: dict[tuple[int, int], int | None] = {k: None for k in keys}
    for i, j in pairs:
        assignment[keys[i]] = j
    used = {j for _, j in pairs}
    shift = float(np.linalg.norm(Q.mean(axis=0) - P.mean(axis=0)))
    info.matched, info.shift_px, info.rotation_deg, info.scale = len(pairs), shift, rotation, scale
    needed = max(1, math.ceil(cfg.min_matched_fraction * len(keys)))
    if len(pairs) < needed:
        info.ok = False
        info.message = f"only {len(pairs)}/{len(keys)} clips fit this part's layout - wrong part or board not in view?"
    elif shift > cfg.max_shift_px:
        info.ok = False
        info.message = f"board shifted {shift:.0f}px (limit {cfg.max_shift_px:.0f}px)"
    else:
        info.message = f"matched {len(pairs)}/{len(keys)} (shift {shift:.0f}px, rotation {rotation:.1f} deg)"
    return Placement(assignment, {k: (float(q[0]), float(q[1])) for k, q in zip(keys, Q)},
                     [j for j in range(len(detections)) if j not in used], info)
