"""Interpret keyboard-wedge scanner input (part number or operator badge)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import BarcodeConfig


@dataclass(frozen=True)
class Scan:
    kind: str  # "part" | "operator" | "unknown"
    value: str
    raw: str


def parse_scan(text: str, cfg: BarcodeConfig, known_parts: set[str] | None = None) -> Scan:
    raw = text
    text = text.strip()
    if cfg.operator_prefix and text.upper().startswith(cfg.operator_prefix.upper()):
        op = text[len(cfg.operator_prefix):].strip()
        return Scan("operator", op, raw) if op else Scan("unknown", text, raw)
    m = re.search(cfg.part_pattern, text)
    if m:
        part = (m.groupdict().get("part") or m.group(0)).strip()
        if known_parts is None or part in known_parts:
            return Scan("part", part, raw)
        # tolerate case differences from scanners configured in another case
        for k in known_parts:
            if k.upper() == part.upper():
                return Scan("part", k, raw)
    return Scan("unknown", text, raw)
