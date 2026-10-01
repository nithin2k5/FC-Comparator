"""Core data model shared by every module.

Labels
------
Every part number has its own label set, created by the person who labels its
images (``red_clip``, ``fork_left``, ``tape`` ...). A label whose name contains
the word ``left`` or ``right`` has a mirror twin (``fork_left`` <-> ``fork_right``):
training adds mirrored images with those labels swapped, and a twin found where
the master expects the other is reported as *wrong orientation*.

Two labels are reserved and never OK: ``missing`` (nothing where the master has
an object) and ``uncertain`` (an object was found but its confidence is below the
threshold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

MISSING = "missing"
UNCERTAIN = "uncertain"

NEVER_OK_LABELS = frozenset({MISSING, UNCERTAIN})
RESERVED_LABELS = NEVER_OK_LABELS

_SIDE = re.compile(r"(?<![A-Za-z])(left|right|LEFT|RIGHT|Left|Right)(?![a-z])")
_LABEL_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class Verdict(str, Enum):
    OK = "OK"
    NG = "NG"


def mirror_name(label: str) -> str:
    """``fork_left`` -> ``fork_right`` (and back); labels without left/right are their own mirror."""
    swap = {"left": "right", "right": "left", "LEFT": "RIGHT", "RIGHT": "LEFT", "Left": "Right", "Right": "Left"}
    return _SIDE.sub(lambda m: swap[m.group(1)], label)


def has_side(label: str) -> bool:
    return mirror_name(label) != label


def check_label(name: str) -> str:
    """Return the cleaned label name or raise ValueError."""
    name = name.strip()
    if not _LABEL_OK.match(name):
        raise ValueError(f"Invalid label {name!r}: use letters, digits, _ and - (no spaces)")
    if name.lower() in RESERVED_LABELS:
        raise ValueError(f"'{name}' is reserved")
    return name


# ---------------------------------------------------------------------------
# Taxonomy: a part's labels plus their mirror twins
# ---------------------------------------------------------------------------
@dataclass
class Taxonomy:
    classes: list[str] = field(default_factory=list)
    mirror: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_labels(cls, labels: list[str]) -> "Taxonomy":
        """The labels, followed by any mirror twin that is not a label itself (learnt from mirrored images)."""
        classes = list(dict.fromkeys(labels))
        for lbl in list(classes):
            twin = mirror_name(lbl)
            if twin != lbl and twin not in classes:
                classes.append(twin)
        return cls(classes, {c: mirror_name(c) for c in classes if has_side(c)})

    def matches(self, expected: str, found: str) -> bool:
        """True when a detected ``found`` satisfies the master's ``expected``."""
        if found in NEVER_OK_LABELS or expected in NEVER_OK_LABELS:
            return False
        return expected == found

    def mirror_of(self, label: str) -> str:
        return self.mirror.get(label, label)


# ---------------------------------------------------------------------------
# Boxes, layouts, part numbers
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """An axis-aligned box in image pixels (top-left x/y, width, height).

    Used both for marked (annotated) clips, confidence 1.0, and for detections.
    """

    label: str
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def iou(self, other: "Box") -> float:
        ax0, ay0, ax1, ay1 = self.xyxy
        bx0, by0, bx1, by1 = other.xyxy
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        d = {"label": self.label, "x": round(self.x, 1), "y": round(self.y, 1), "w": round(self.w, 1), "h": round(self.h, 1)}
        if self.confidence != 1.0:
            d["confidence"] = round(self.confidence, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(str(d["label"]), float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]), float(d.get("confidence", 1.0)))


@dataclass
class Layout:
    """Expected clip positions of a part, learned from its marked master image.

    ``positions[(cable, row)] = (x, y)`` are clip centres in master-image pixels.
    """

    cables: int
    rows: int
    positions: dict[tuple[int, int], tuple[float, float]]
    clip_size: tuple[float, float]
    image_size: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "cables": self.cables,
            "rows": self.rows,
            "clip_size": [round(self.clip_size[0], 1), round(self.clip_size[1], 1)],
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "positions": [[c, r, round(x, 1), round(y, 1)] for (c, r), (x, y) in sorted(self.positions.items())],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(
            cables=int(d["cables"]),
            rows=int(d["rows"]),
            positions={(int(c), int(r)): (float(x), float(y)) for c, r, x, y in d["positions"]},
            clip_size=(float(d["clip_size"][0]), float(d["clip_size"][1])),
            image_size=(int(d["image_size"][0]), int(d["image_size"][1])),
        )


@dataclass
class PartNumber:
    """A part's master: the label expected at every position of a known-good board.

    ``pattern[row - 1][cable - 1]`` is the expected label; rows run top to bottom,
    cables left to right. ``layout`` says where each position sits in the master photo.
    """

    code: str
    pattern: list[list[str]]
    description: str = ""
    layout: Layout | None = None
    master_image: str = ""  # id of the master photo among the part's images

    @property
    def rows(self) -> int:
        return len(self.pattern)

    @property
    def cables(self) -> int:
        return len(self.pattern[0]) if self.pattern else 0

    def expected(self, cable: int, row: int) -> str:
        return self.pattern[row - 1][cable - 1]

    def labels(self) -> set[str]:
        return {p for row in self.pattern for p in row}

    def validate(self, taxonomy: Taxonomy | None = None) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("Part number code must not be empty")
        if not self.pattern or not self.pattern[0]:
            raise ValueError(f"{self.code}: pattern is empty")
        if any(len(row) != self.cables for row in self.pattern):
            raise ValueError(f"{self.code}: every row must list one object per cable ({self.cables})")
        if taxonomy is not None:
            bad = sorted(self.labels() - set(taxonomy.classes))
            if bad:
                raise ValueError(f"{self.code}: master uses unknown labels {bad}")
        if self.layout is not None and (self.layout.rows != self.rows or self.layout.cables != self.cables):
            raise ValueError(
                f"{self.code}: layout is {self.layout.cables}x{self.layout.rows} but part is {self.cables}x{self.rows}"
            )

    def to_dict(self) -> dict:
        d: dict = {"pattern": [list(row) for row in self.pattern]}
        if self.master_image:
            d["master_image"] = self.master_image
        if self.layout is not None:
            d["layout"] = self.layout.to_dict()
        return d

    @classmethod
    def from_dict(cls, code: str, d: dict, description: str = "") -> "PartNumber":
        d = d or {}
        return cls(
            code=str(code),
            pattern=[[str(p) for p in row] for row in d.get("pattern", [])],
            description=description,
            layout=Layout.from_dict(d["layout"]) if d.get("layout") else None,
            master_image=str(d.get("master_image", "") or ""),
        )


# ---------------------------------------------------------------------------
# Inspection results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Classification:
    """What was found at one position."""

    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    guess: str = ""  # detector's label when ``label`` was downgraded to uncertain


@dataclass(frozen=True)
class PositionResult:
    cable: int
    row: int
    expected: str
    found: str
    confidence: float
    ok: bool
    reason: str = ""
    box: tuple[float, float, float, float] | None = None  # x, y, w, h in the inspected image

    @property
    def key(self) -> tuple[int, int]:
        return (self.cable, self.row)


@dataclass
class PlacementInfo:
    """How the detected clips were fitted to the part's layout."""

    ok: bool = True
    message: str = ""
    shift_px: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    matched: int = 0
    expected: int = 0


@dataclass
class InspectionReport:
    part_number: str
    mode: str
    positions: list[PositionResult]
    placement: PlacementInfo = field(default_factory=PlacementInfo)
    extras: list[Box] = field(default_factory=list)  # confident clips where the master has none
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    operator_id: str = ""
    detector: str = ""
    error: str = ""

    @property
    def verdict(self) -> Verdict:
        if self.error or not self.positions or self.extras:
            return Verdict.NG
        return Verdict.OK if all(p.ok for p in self.positions) else Verdict.NG

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK

    @property
    def mismatches(self) -> list[PositionResult]:
        return [p for p in self.positions if not p.ok]

