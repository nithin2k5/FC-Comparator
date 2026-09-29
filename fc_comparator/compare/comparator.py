"""Compare classified positions against the master pattern and/or each other.

Modes
-----
master (A, default)
    Every cable's clip at row r must match ``pattern[r]``.
cross (B)
    All cables in the same row must match each other. The row's majority label
    becomes the reference; cables that differ are flagged. Without a strict
    majority (e.g. 2-2) every cable in the row is flagged. Missing/uncertain
    can never become the majority reference.
both
    A position is OK only if it passes both checks.

Missing and uncertain positions are never OK in any mode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ..models import (
    MISSING,
    NEVER_OK_LABELS,
    UNCERTAIN,
    Classification,
    PartNumber,
    PositionResult,
    base_type,
    label_matches,
)

MODES = ("master", "cross", "both")


def compare(
    found: Mapping[tuple[int, int], Classification],
    part: PartNumber | None,
    cables: int,
    rows: int,
    mode: str = "master",
) -> list[PositionResult]:
    """Return one PositionResult per (cable, row), ordered by cable then row.

    ``found`` maps (cable, row) -> Classification whose label may already be
    ``uncertain``. A position absent from ``found`` is treated as uncertain.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown compare mode {mode!r}; expected one of {MODES}")
    if mode in ("master", "both"):
        if part is None:
            raise ValueError(f"Mode {mode!r} needs a part number")
        part.validate(rows)

    def get(c: int, r: int) -> Classification:
        return found.get((c, r)) or Classification(UNCERTAIN, 0.0)

    master = _master(get, part, cables, rows) if mode in ("master", "both") else None
    cross = _cross(get, cables, rows) if mode in ("cross", "both") else None

    if mode == "master":
        return master
    if mode == "cross":
        return cross
    return [_merge(a, b) for a, b in zip(master, cross)]


def _master(get, part: PartNumber, cables: int, rows: int) -> list[PositionResult]:
    results = []
    for c in range(1, cables + 1):
        for r in range(1, rows + 1):
            cls = get(c, r)
            expected = part.pattern[r - 1]
            ok = label_matches(expected, cls.label)
            results.append(
                PositionResult(c, r, expected, cls.label, cls.confidence, ok, "" if ok else _reason(expected, cls.label))
            )
    return results


def _cross(get, cables: int, rows: int) -> list[PositionResult]:
    per_row: dict[int, list[PositionResult]] = {}
    for r in range(1, rows + 1):
        labels = {c: get(c, r) for c in range(1, cables + 1)}
        majority = _majority([cls.label for cls in labels.values()])
        row_results = []
        for c, cls in labels.items():
            if majority is None:
                ok, expected, reason = False, "?", "no majority in row"
                if cls.label in NEVER_OK_LABELS:
                    reason = _reason("?", cls.label)
            else:
                expected = majority
                ok = cls.label not in NEVER_OK_LABELS and _same(majority, cls.label)
                reason = "" if ok else _reason(majority, cls.label) + " (odd one out)"
            row_results.append(PositionResult(c, r, expected, cls.label, cls.confidence, ok, reason))
        per_row[r] = row_results
    # order by cable, then row, like master mode
    return sorted((p for rs in per_row.values() for p in rs), key=lambda p: (p.cable, p.row))


def _majority(labels: list[str]) -> str | None:
    """Label held by more than half of the cables; None if there is none or it is missing/uncertain."""
    if not labels:
        return None
    label, count = Counter(labels).most_common(1)[0]
    if count * 2 <= len(labels) or label in NEVER_OK_LABELS:
        return None
    return label


def _same(reference: str, found: str) -> bool:
    if reference == found:
        return True
    # An orientation-less "fork" is compatible with an oriented one at the type level only
    # when either side lacks orientation.
    if "fork" in (reference, found) and base_type(reference) == base_type(found):
        return True
    return False


def _merge(a: PositionResult, b: PositionResult) -> PositionResult:
    ok = a.ok and b.ok
    reasons = "; ".join(x for x in (a.reason, b.reason) if x)
    return PositionResult(a.cable, a.row, a.expected, a.found, a.confidence, ok, reasons)


def _reason(expected: str, found: str) -> str:
    if found == MISSING:
        return "clip missing"
    if found == UNCERTAIN:
        return "low confidence"
    if base_type(expected) == base_type(found) == "fork":
        return f"fork orientation {found} != {expected}"
    return f"{found} != {expected}"
