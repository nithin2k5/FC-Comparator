"""Draw inspection results on the board image (for the UI and saved evidence)."""

from __future__ import annotations

import cv2
import numpy as np

from .models import MISSING, InspectionReport

GREEN = (40, 170, 40)
RED = (30, 30, 220)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def short(label: str) -> str:
    return {"fork_left": "fork-L", "fork_right": "fork-R", "uncertain": "?"}.get(label, label)


def _text_box(img, text_parts, org, scale, color, thickness=2):
    """Draw text on a filled background. The token NEQ draws a real "≠"."""
    widths = [int(22 * scale) if p == "NEQ" else cv2.getTextSize(p, FONT, scale, thickness)[0][0] for p in text_parts]
    th = cv2.getTextSize("Ag", FONT, scale, thickness)[0][1]
    gap = int(6 * scale)
    total = sum(widths) + gap * (len(text_parts) - 1)
    x = max(0, min(int(org[0]), img.shape[1] - total - 8))
    y = max(th + 8, int(org[1]))
    cv2.rectangle(img, (x, y - th - 8), (x + total + 8, y + 6), color, -1)
    cx = x + 4
    for part, w in zip(text_parts, widths):
        if part == "NEQ":  # Hershey fonts have no "≠": draw "=" and strike it through
            mid, gy = y - th // 2, max(3, th // 4)
            cv2.line(img, (cx, mid - gy), (cx + w, mid - gy), WHITE, thickness, cv2.LINE_AA)
            cv2.line(img, (cx, mid + gy), (cx + w, mid + gy), WHITE, thickness, cv2.LINE_AA)
            cv2.line(img, (cx + w, mid - 2 * gy - 2), (cx, mid + 2 * gy + 2), WHITE, thickness, cv2.LINE_AA)
        else:
            cv2.putText(img, part, (cx, y), FONT, scale, WHITE, thickness, cv2.LINE_AA)
        cx += w + gap


def _dashed_rect(img, p0, p1, color, thickness=3, dash=12):
    (x0, y0), (x1, y1) = p0, p1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, length, dash * 2):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            pa = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            pb = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def annotate(image: np.ndarray, report: InspectionReport, header: bool = True) -> np.ndarray:
    out = image.copy()
    scale = max(0.5, out.shape[0] / 1600)
    pad = 6
    for p in report.positions:
        if p.box is None:
            continue
        x, y, w, h = p.box
        p0, p1 = (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad))
        tag = f"C{p.cable}R{p.row}"
        if p.ok:
            cv2.rectangle(out, p0, p1, GREEN, 3)
            _text_box(out, [f"{tag} {short(p.found)} {p.confidence:.0%}"], (p0[0], p1[1] + int(26 * scale)), scale * 0.75, GREEN, 1)
        elif p.found == MISSING:
            _dashed_rect(out, p0, p1, RED, 4)
            _text_box(out, [tag, "missing", short(p.expected)], (p0[0], p0[1] - 4), scale, RED)
        else:
            cv2.rectangle(out, p0, p1, RED, 5)
            _text_box(out, [tag, short(p.found), "NEQ", short(p.expected)], (p0[0], p0[1] - 4), scale, RED)
            _text_box(out, [f"{p.confidence:.0%}"], (p0[0], p1[1] + int(26 * scale)), scale * 0.75, RED, 1)
    for b in report.extras:
        p0, p1 = (int(b.x - pad), int(b.y - pad)), (int(b.x + b.w + pad), int(b.y + b.h + pad))
        cv2.rectangle(out, p0, p1, ORANGE, 5)
        _text_box(out, [f"unexpected {short(b.label)} {b.confidence:.0%}"], (p0[0], p0[1] - 4), scale, ORANGE)
    return _header(out, report) if header else out


def _header(img: np.ndarray, report: InspectionReport) -> np.ndarray:
    bar_h = max(40, img.shape[0] // 18)
    color = GREEN if report.ok else RED
    bar = np.full((bar_h, img.shape[1], 3), color, np.uint8)
    s = bar_h / 45
    text = f"{report.verdict.value}  {report.part_number}  {report.timestamp:%Y-%m-%d %H:%M:%S}"
    if not report.ok:
        n = len(report.mismatches) + len(report.extras)
        text += f"  {n} finding(s)" if n else ""
    if report.error:
        text += f"  [{report.error[:70]}]"
    if report.operator_id:
        text += f"  op {report.operator_id}"
    cv2.putText(bar, text, (12, int(bar_h * 0.72)), FONT, s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    return np.vstack([bar, img])


def draw_boxes(image: np.ndarray, boxes, colors: dict[str, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Plain box overlay (label + confidence), e.g. to preview detections."""
    out = image.copy()
    scale = max(0.45, out.shape[0] / 1800)
    for b in boxes:
        c = (colors or {}).get(b.label, (255, 120, 0))
        cv2.rectangle(out, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), c, 2)
        _text_box(out, [f"{short(b.label)} {b.confidence:.0%}"], (b.x, b.y - 3), scale, c, 1)
    return out
