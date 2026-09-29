"""Draw inspection results on the board image (for the UI and saved evidence)."""

from __future__ import annotations

import cv2
import numpy as np

from .models import InspectionReport, Roi

GREEN = (40, 170, 40)
RED = (30, 30, 220)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def short(label: str) -> str:
    return {"fork_left": "fork-L", "fork_right": "fork-R", "uncertain": "?"}.get(label, label)


def _text_box(img, text_parts, org, scale, color, thickness=2):
    """Draw text on a filled background. ``text_parts`` may contain the token NEQ for a real "≠"."""
    widths = []
    for part in text_parts:
        if part == "NEQ":
            widths.append(int(22 * scale))
        else:
            widths.append(cv2.getTextSize(part, FONT, scale, thickness)[0][0])
    th = cv2.getTextSize("Ag", FONT, scale, thickness)[0][1]
    gap = int(6 * scale)
    total = sum(widths) + gap * (len(text_parts) - 1)
    x, y = org
    x = max(0, min(x, img.shape[1] - total - 8))
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + total + 8, y + 6), color, -1)
    cx = x + 4
    for part, w in zip(text_parts, widths):
        if part == "NEQ":  # Hershey fonts have no "≠": draw "=" and strike it through
            mid, gap_y = y - th // 2, max(3, th // 4)
            cv2.line(img, (cx, mid - gap_y), (cx + w, mid - gap_y), WHITE, thickness, cv2.LINE_AA)
            cv2.line(img, (cx, mid + gap_y), (cx + w, mid + gap_y), WHITE, thickness, cv2.LINE_AA)
            cv2.line(img, (cx + w, mid - 2 * gap_y - 2), (cx, mid + 2 * gap_y + 2), WHITE, thickness, cv2.LINE_AA)
        else:
            cv2.putText(img, part, (cx, y), FONT, scale, WHITE, thickness, cv2.LINE_AA)
        cx += w + gap


def annotate(image: np.ndarray, report: InspectionReport, rois: list[Roi], header: bool = True) -> np.ndarray:
    out = image.copy()
    h = out.shape[0]
    scale = max(0.5, h / 1600)
    by_key = {r.key: r for r in rois}
    for p in report.positions:
        roi = by_key.get(p.key)
        if roi is None:
            continue
        color = GREEN if p.ok else RED
        cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 5 if not p.ok else 3)
        if p.ok:
            _text_box(out, [f"{short(p.found)} {p.confidence:.0%}"], (roi.x, roi.y + roi.h + int(28 * scale)), scale * 0.8, color, 1)
        else:
            _text_box(out, [short(p.found), "NEQ", short(p.expected)], (roi.x, roi.y - 4), scale, color)
            _text_box(out, [f"{p.confidence:.0%}"], (roi.x, roi.y + roi.h + int(28 * scale)), scale * 0.8, color, 1)
    if header:
        out = _header(out, report)
    return out


def _header(img: np.ndarray, report: InspectionReport) -> np.ndarray:
    bar_h = max(40, img.shape[0] // 18)
    color = GREEN if report.ok else RED
    bar = np.full((bar_h, img.shape[1], 3), color, np.uint8)
    s = bar_h / 45
    text = f"{report.verdict.value}  {report.part_number}  {report.timestamp:%Y-%m-%d %H:%M:%S}"
    if not report.ok:
        text += f"  {len(report.mismatches)} NG position(s)"
    if report.error:
        text += f"  [{report.error}]"
    if report.operator_id:
        text += f"  op {report.operator_id}"
    cv2.putText(bar, text, (12, int(bar_h * 0.72)), FONT, s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    return np.vstack([bar, img])
