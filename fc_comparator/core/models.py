"""Core data model shared by every module.

Clip types (taxonomy)
---------------------
The detector classes are configurable (``Taxonomy.classes``) because new part
numbers can bring new clip types. Defaults are the clips on the FCC boards:
``round``, ``small`` and three fork brackets - black ``fork``, ``grey_fork`` and
``metal_fork`` - each with the side its open end faces (``_left`` / ``_right``).

* **groups** - an expected label that accepts several detected classes, e.g.
  ``fork`` accepts ``fork_left`` and ``fork_right`` (orientation not checked).
* **mirror** - classes that are horizontal mirror images of each other; used to
  augment training data and templates (a marked ``fork_left`` also teaches
  ``fork_right``).

Two labels are reserved and never OK: ``missing`` (no clip where the master has
one) and ``uncertain`` (a clip was found but its confidence is below the
threshold).
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
FORK_KINDS = ("fork", "grey_fork", "metal_fork")
MISSING = "missing"
UNCERTAIN = "uncertain"

NEVER_OK_LABELS = frozenset({MISSING, UNCERTAIN})
RESERVED_LABELS = NEVER_OK_LABELS


class Verdict(str, Enum):
    OK = "OK"
    NG = "NG"


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
@dataclass
class Taxonomy:
    classes: list[str] = field(
        default_factory=lambda: [ROUND, SMALL] + [f"{k}_{side}" for k in FORK_KINDS for side in ("left", "right")])
    groups: dict[str, list[str]] = field(
        default_factory=lambda: {k: [f"{k}_left", f"{k}_right"] for k in FORK_KINDS})
    mirror: dict[str, str] = field(
        default_factory=lambda: {f"{k}_{a}": f"{k}_{b}" for k in FORK_KINDS for a, b in (("left", "right"), ("right", "left"))})

    def expected_labels(self) -> list[str]:
        """Labels allowed in a master pattern: every class plus every group name."""
        return list(self.classes) + [g for g in self.groups if g not in self.classes]

    def matches(self, expected: str, found: str) -> bool:
        """True when a detected ``found`` satisfies the master's ``expected``."""
        if found in NEVER_OK_LABELS or expected in NEVER_OK_LABELS:
            return False
        if expected in self.groups:
            return found == expected or found in self.groups[expected]
        return expected == found

    def group_of(self, label: str) -> str:
        for name, members in self.groups.items():
            if label in members:
                return name
        return label

    def mirror_of(self, label: str) -> str:
        return self.mirror.get(label, label)

    def validate(self) -> list[str]:
        problems = []
        if not self.classes:
            problems.append("taxonomy.classes is empty")
        if len(set(self.classes)) != len(self.classes):
            problems.append("taxonomy.classes contains duplicates")
        for name in self.classes:
            if name in RESERVED_LABELS:
                problems.append(f"'{name}' is reserved and cannot be a clip class")
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                problems.append(f"invalid class name {name!r} (letters, digits, _ and - only)")
        for g, members in self.groups.items():
            unknown = [m for m in members if m not in self.classes]
            if unknown:
                problems.append(f"group '{g}' refers to unknown classes {unknown}")
        for a, b in self.mirror.items():
            if a not in self.classes or b not in self.classes:
                problems.append(f"mirror pair {a} <-> {b} uses unknown classes")
            elif self.mirror.get(b) != a:
                problems.append(f"mirror pair {a} -> {b} is not symmetric")
        return problems

    def to_dict(self) -> dict:
        return {"classes": list(self.classes), "groups": {k: list(v) for k, v in self.groups.items()},
                "mirror": dict(self.mirror)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Taxonomy":
        if not d:
            return cls()
        t = cls()
        if "classes" in d:
            t.classes = [str(c) for c in d["classes"]]
        if "groups" in d:
            t.groups = {str(k): [str(x) for x in v] for k, v in (d["groups"] or {}).items()}
        if "mirror" in d:
            t.mirror = {str(k): str(v) for k, v in (d["mirror"] or {}).items()}
        return t


DEFAULT_TAXONOMY = Taxonomy()


def label_matches(expected: str, found: str, taxonomy: Taxonomy = DEFAULT_TAXONOMY) -> bool:
    return taxonomy.matches(expected, found)


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
    """A part number: the clip expected at every position of the board.

    ``pattern[row - 1][cable - 1]`` is the expected label (a clip class or a group
    name such as ``fork``); rows run top to bottom, cables left to right.
    """

    code: str
    pattern: list[list[str]]
    description: str = ""
    layout: Layout | None = None
    master_image: str = ""  # data-set image id the master was created from

    @property
    def rows(self) -> int:
        return len(self.pattern)

    @property
    def cables(self) -> int:
        return len(self.pattern[0]) if self.pattern else 0

    def expected(self, cable: int, row: int) -> str:
        return self.pattern[row - 1][cable - 1]

    def validate(self, taxonomy: Taxonomy = DEFAULT_TAXONOMY) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("Part number code must not be empty")
        if not self.pattern or not self.pattern[0]:
            raise ValueError(f"{self.code}: pattern is empty")
        if any(len(row) != self.cables for row in self.pattern):
            raise ValueError(f"{self.code}: every row must list one clip per cable ({self.cables})")
        allowed = taxonomy.expected_labels()
        bad = sorted({p for row in self.pattern for p in row if p not in allowed})
        if bad:
            raise ValueError(f"{self.code}: invalid pattern labels {bad}; allowed: {allowed}")
        if self.layout is not None and (self.layout.rows != self.rows or self.layout.cables != self.cables):
            raise ValueError(
                f"{self.code}: layout is {self.layout.cables}x{self.layout.rows} but part is {self.cables}x{self.rows}"
            )

    def to_dict(self) -> dict:
        d: dict = {"description": self.description, "pattern": [list(row) for row in self.pattern]}
        if self.master_image:
            d["master_image"] = self.master_image
        if self.layout is not None:
            d["layout"] = self.layout.to_dict()
        return d

    @classmethod
    def from_dict(cls, code: str, d: dict | list) -> "PartNumber":
        if isinstance(d, list):  # shorthand: P001: [[round, round], [fork, fork]]
            d = {"pattern": d}
        d = d or {}
        raw = d.get("pattern", [])
        if raw and all(isinstance(p, str) for p in raw):  # one label per row, the same on every cable
            raw = [[p] * int(d.get("cables", 4)) for p in raw]
        return cls(
            code=str(code),
            pattern=[[str(p) for p in row] for row in raw],
            description=str(d.get("description", "") or ""),
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

