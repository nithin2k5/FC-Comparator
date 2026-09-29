"""Compare what was found at each position against the master pattern and/or each other.

Modes
-----
master (A, default)
    Every cable's clip at row r must match ``pattern[r]`` (a group name such as
    ``fork`` accepts any of its member classes).
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
    DEFAULT_TAXONOMY,
    MISSING,
    NEVER_OK_LABELS,
    UNCERTAIN,
    Classification,
    PartNumber,
    PositionResult,
    Taxonomy,
)

MODES = ("master", "cross", "both")


def compare(
    found: Mapping[tuple[int, int], Classification],
    part: PartNumber | None,
    cables: int,
    rows: int,
    mode: str = "master",
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
) -> list[PositionResult]:
    """Return one PositionResult per (cable, row), ordered by cable then row.

    ``found`` maps (cable, row) -> Classification whose label may be ``missing``
    or ``uncertain``. A position absent from ``found`` is treated as uncertain.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown compare mode {mode!r}; expected one of {MODES}")
    if mode in ("master", "both"):
        if part is None:
            raise ValueError(f"Mode {mode!r} needs a part number")
        part.validate(taxonomy)
        if part.rows != rows:
            raise ValueError(f"{part.code}: pattern has {part.rows} rows, board has {rows}")

    def get(c: int, r: int) -> Classification:
        return found.get((c, r)) or Classification(UNCERTAIN, 0.0)

    master = _master(get, part, cables, rows, taxonomy) if mode in ("master", "both") else None
    cross = _cross(get, cables, rows) if mode in ("cross", "both") else None

    if mode == "master":
        return master
    if mode == "cross":
        return cross
    return [_merge(a, b) for a, b in zip(master, cross)]


def _master(get, part: PartNumber, cables: int, rows: int, taxonomy: Taxonomy) -> list[PositionResult]:
    results = []
    for c in range(1, cables + 1):
        for r in range(1, rows + 1):
            cls = get(c, r)
            expected = part.pattern[r - 1]
            ok = taxonomy.matches(expected, cls.label)
            reason = "" if ok else _reason(expected, cls.label, taxonomy)
            results.append(PositionResult(c, r, expected, cls.label, cls.confidence, ok, reason))
    return results


def _cross(get, cables: int, rows: int) -> list[PositionResult]:
    results = []
    for r in range(1, rows + 1):
        labels = {c: get(c, r) for c in range(1, cables + 1)}
        majority = _majority([cls.label for cls in labels.values()])
        for c, cls in labels.items():
            if majority is None:
                reason = "no majority in row" if cls.label not in NEVER_OK_LABELS else _reason("?", cls.label, None)
                results.append(PositionResult(c, r, "?", cls.label, cls.confidence, False, reason))
            else:
                ok = cls.label not in NEVER_OK_LABELS and cls.label == majority
                reason = "" if ok else _reason(majority, cls.label, None) + " (odd one out)"
                results.append(PositionResult(c, r, majority, cls.label, cls.confidence, ok, reason))
    return sorted(results, key=lambda p: (p.cable, p.row))


def _majority(labels: list[str]) -> str | None:
    """Label held by more than half of the cables; None if there is none or it is missing/uncertain."""
    if not labels:
        return None
    label, count = Counter(labels).most_common(1)[0]
    if count * 2 <= len(labels) or label in NEVER_OK_LABELS:
        return None
    return label


def _merge(a: PositionResult, b: PositionResult) -> PositionResult:
    reasons = "; ".join(x for x in (a.reason, b.reason) if x)
    return PositionResult(a.cable, a.row, a.expected, a.found, a.confidence, a.ok and b.ok, reasons)


def _reason(expected: str, found: str, taxonomy: Taxonomy | None) -> str:
    if found == MISSING:
        return "clip missing"
    if found == UNCERTAIN:
        return "low confidence"
    if taxonomy is not None and taxonomy.mirror_of(found) != found:
        if expected == taxonomy.mirror_of(found):
            return f"wrong orientation: {found} != {expected}"
    return f"{found} != {expected}"
