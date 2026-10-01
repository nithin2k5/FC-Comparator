"""Compare what was found at each position with the part's master.

Every position (cable, row) must hold the label the master expects there.
Missing and uncertain positions are never OK; a left/right twin of the expected
label is reported as wrong orientation.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import (
    MISSING,
    UNCERTAIN,
    Classification,
    PartNumber,
    PositionResult,
    Taxonomy,
)


def compare(
    found: Mapping[tuple[int, int], Classification],
    part: PartNumber,
    taxonomy: Taxonomy | None = None,
) -> list[PositionResult]:
    """One PositionResult per (cable, row), ordered by cable then row.

    ``found`` maps (cable, row) -> Classification whose label may be ``missing``
    or ``uncertain``. A position absent from ``found`` is treated as uncertain.
    """
    taxonomy = taxonomy or Taxonomy.from_labels(sorted(part.labels()))
    part.validate()
    results = []
    for c in range(1, part.cables + 1):
        for r in range(1, part.rows + 1):
            cls = found.get((c, r)) or Classification(UNCERTAIN, 0.0)
            expected = part.expected(c, r)
            ok = taxonomy.matches(expected, cls.label)
            results.append(PositionResult(c, r, expected, cls.label, cls.confidence, ok,
                                          "" if ok else reason(expected, cls.label, taxonomy)))
    return results


def reason(expected: str, found: str, taxonomy: Taxonomy) -> str:
    if found == MISSING:
        return "object missing"
    if found == UNCERTAIN:
        return "low confidence"
    mirrored = taxonomy.mirror_of(found)
    if mirrored != found and taxonomy.matches(expected, mirrored):
        return f"wrong orientation: {found} != {expected}"
    return f"{found} != {expected}"
