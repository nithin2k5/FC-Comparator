"""Core data model shared by every module.

Labels
------
Classifier output labels (what a crop *is*):
    round, fork_left, fork_right, fork, small, missing
``fork`` without orientation is produced by models trained without
orientation-specific classes. ``uncertain`` is assigned by the pipeline when
confidence falls below the threshold.

Expected labels (what a master pattern *requires*):
    round, fork, fork_left, fork_right, small
``fork`` accepts either orientation; ``fork_left``/``fork_right`` require it.
``missing`` and ``uncertain`` are never valid expectations and never OK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

ROUND = "round"
FORK = "fork"
FORK_LEFT = "fork_left"
FORK_RIGHT = "fork_right"
SMALL = "small"
MISSING = "missing"
UNCERTAIN = "uncertain"

CLASSIFIER_LABELS = (ROUND, FORK, FORK_LEFT, FORK_RIGHT, SMALL, MISSING)
EXPECTED_LABELS = (ROUND, FORK, FORK_LEFT, FORK_RIGHT, SMALL)
NEVER_OK_LABELS = frozenset({MISSING, UNCERTAIN})

_BASE_TYPE = {FORK_LEFT: FORK, FORK_RIGHT: FORK}


def base_type(label: str) -> str:
    """Collapse orientation: fork_left/fork_right -> fork."""
    return _BASE_TYPE.get(label, label)


def label_matches(expected: str, found: str) -> bool:
    """True when ``found`` satisfies ``expected``.

    Missing/uncertain never satisfy anything. A plain ``fork`` expectation
    accepts any fork; an oriented expectation needs the exact orientation, so an
    orientation-less ``fork`` detection does not satisfy ``fork_left``.
    """
    if found in NEVER_OK_LABELS or expected in NEVER_OK_LABELS:
        return False
    if expected == FORK:
        return base_type(found) == FORK
    return expected == found


def mirror_label(label: str) -> str:
    """Label of the horizontally mirrored clip."""
    return {FORK_LEFT: FORK_RIGHT, FORK_RIGHT: FORK_LEFT}.get(label, label)


class Verdict(str, Enum):
    OK = "OK"
    NG = "NG"


@dataclass(frozen=True)
class Roi:
    """Inspection position in reference-image pixel coordinates (1-based cable/row)."""

    cable: int
    row: int
    x: int
    y: int
    w: int
    h: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.cable, self.row)

    def to_dict(self) -> dict:
        return {"cable": self.cable, "row": self.row, "x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class PartNumber:
    """A part number and its master pattern (one expected label per row, top to bottom)."""

    code: str
    pattern: list[str]
    description: str = ""

    def validate(self, rows: int) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("Part number code must not be empty")
        if len(self.pattern) != rows:
            raise ValueError(f"{self.code}: pattern has {len(self.pattern)} entries, station has {rows} rows")
        bad = [p for p in self.pattern if p not in EXPECTED_LABELS]
        if bad:
            raise ValueError(f"{self.code}: invalid pattern labels {bad}; allowed: {list(EXPECTED_LABELS)}")

    def to_dict(self) -> dict:
        return {"description": self.description, "pattern": list(self.pattern)}


@dataclass(frozen=True)
class Classification:
    """Raw classifier output for one crop."""

    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    guess: str = ""  # classifier's best label when ``label`` was downgraded to uncertain


@dataclass(frozen=True)
class PositionResult:
    cable: int
    row: int
    expected: str
    found: str
    confidence: float
    ok: bool
    reason: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return (self.cable, self.row)


@dataclass
class AlignmentInfo:
    applied: bool = False
    ok: bool = True
    inliers: int = 0
    shift_px: float = 0.0
    message: str = ""


@dataclass
class InspectionReport:
    part_number: str
    mode: str
    positions: list[PositionResult]
    alignment: AlignmentInfo = field(default_factory=AlignmentInfo)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    operator_id: str = ""
    classifier: str = ""
    error: str = ""

    @property
    def verdict(self) -> Verdict:
        if self.error or not self.positions:
            return Verdict.NG
        return Verdict.OK if all(p.ok for p in self.positions) else Verdict.NG

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK

    @property
    def mismatches(self) -> list[PositionResult]:
        return [p for p in self.positions if not p.ok]
