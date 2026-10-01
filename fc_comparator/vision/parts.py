"""Part numbers on disk: every part has its own images, labels, trained models and master.

Layout::

    <parts_dir>/<code>/part.json              description, active model, master (pattern + layout)
    <parts_dir>/<code>/index.json             label names + the boxes labelled on every image
    <parts_dir>/<code>/images/<id>.jpg        the part's images
    <parts_dir>/<code>/models/v001.pt         trained model versions (all kept, for rollback)
    <parts_dir>/<code>/models/v001.json       its evaluation report (accuracy, per label, when, ...)
    <parts_dir>/<code>/models/v001_confusion.png
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core.models import PartNumber, Taxonomy
from .dataset import AnnotationStore

PART_FILE = "part.json"
_CODE_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION = re.compile(r"^v(\d{3,})$")


def check_code(code: str) -> str:
    code = code.strip()
    if not _CODE_OK.match(code) or len(code) > 64:
        raise ValueError(f"Invalid part number {code!r}: use letters, digits, '.', '_' and '-'")
    return code


@dataclass
class ModelInfo:
    """One trained model version of a part."""

    version: str
    path: Path
    report: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        acc = self.report.get("accuracy")
        return float(acc) if acc is not None else None

    @property
    def trained(self) -> str:
        return str(self.report.get("trained", ""))

    @property
    def images(self) -> int:
        return int(self.report.get("labelled_images", 0) or 0)

    @property
    def classes(self) -> list[str]:
        return list(self.report.get("classes", []))

    def meets(self, target: float) -> bool:
        return self.accuracy is not None and self.accuracy >= target

    def weak_labels(self, target: float) -> list[tuple[str, float, float]]:
        """Labels whose precision or recall on the held-back images is below ``target``: (label, precision, recall)."""
        out = []
        for label, (p, r) in (self.report.get("precision_recall") or {}).items():
            if p < target or r < target:
                out.append((label, float(p), float(r)))
        return sorted(out, key=lambda t: min(t[1], t[2]))

    def summary(self) -> str:
        acc = f"{self.accuracy:.1%}" if self.accuracy is not None else "not scored"
        when = self.trained.replace("T", " ")[:16]
        return f"{self.version} ({acc}, trained {when or '?'}, {self.images} images)"


class Part:
    """One part number's folder."""

    def __init__(self, root: Path, code: str):
        self.code = code
        self.dir = Path(root) / code
        self._lock = threading.RLock()
        self.description = ""
        self.created = ""
        self.active_model = ""
        self.master: PartNumber | None = None
        self._store: AnnotationStore | None = None
        self.reload()

    # -- persistence -----------------------------------------------------------
    def reload(self) -> None:
        with self._lock:
            path = self.dir / PART_FILE
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            self.description = str(data.get("description", "") or "")
            self.created = str(data.get("created", "") or "")
            self.active_model = str(data.get("active_model", "") or "")
            m = data.get("master")
            self.master = PartNumber.from_dict(self.code, m, self.description) if m and m.get("pattern") else None
            if self._store is not None:
                self._store.reload()

    def save(self) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            data = {"code": self.code, "description": self.description, "created": self.created,
                    "active_model": self.active_model, "master": self.master.to_dict() if self.master else None}
            fd, tmp = tempfile.mkstemp(prefix=".part-", suffix=".json", dir=self.dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=1)
                os.replace(tmp, self.dir / PART_FILE)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # -- images and labels -------------------------------------------------------
    @property
    def store(self) -> AnnotationStore:
        if self._store is None:
            self._store = AnnotationStore(self.dir)
        return self._store

    @property
    def labels(self) -> list[str]:
        return self.store.labels

    def taxonomy(self) -> Taxonomy:
        """The part's labels (+ mirror twins); the master's labels too, in case one was renamed since."""
        extra = sorted(self.master.labels() - set(self.labels)) if self.master else []
        return Taxonomy.from_labels(self.labels + extra)

    def labelled_images(self) -> int:
        return sum(r.marked for r in self.store.records())

    def rename_label(self, old: str, new: str) -> int:
        """Rename (or merge into an existing label) on every image and in the master."""
        n = self.store.rename_label(old, new)
        if self.master is not None and old in self.master.labels():
            self.master.pattern = [[new if p == old else p for p in row] for row in self.master.pattern]
            self.save()
        return n

    def delete_label(self, name: str) -> int:
        if self.master is not None and name in self.master.labels():
            raise ValueError(f"'{name}' is used by the master. Set the master again without it first.")
        return self.store.delete_label(name)

    # -- models ----------------------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.dir / "models"

    def models(self) -> list[ModelInfo]:
        """All trained versions, oldest first."""
        out = []
        if self.models_dir.is_dir():
            for p in sorted(self.models_dir.glob("v*.pt")):
                if _VERSION.match(p.stem):
                    out.append(ModelInfo(p.stem, p, _read_json(p.with_suffix(".json"))))
        return out

    def model(self, version: str) -> ModelInfo | None:
        return next((m for m in self.models() if m.version == version), None)

    def active(self) -> ModelInfo | None:
        return self.model(self.active_model) if self.active_model else None

    def next_version(self) -> str:
        nums = [int(_VERSION.match(m.version).group(1)) for m in self.models()]
        return f"v{max(nums, default=0) + 1:03d}"

    def set_active(self, version: str) -> None:
        if self.model(version) is None:
            raise KeyError(f"{self.code} has no model {version}")
        self.active_model = version
        self.save()

    # -- master ----------------------------------------------------------------------
    def set_master(self, master: PartNumber) -> None:
        master.description = self.description
        self.master = master
        self.save()

    # -- readiness -------------------------------------------------------------------
    def setup_problems(self, backend: str = "yolo") -> list[str]:
        """Why this part cannot be inspected yet (empty = ready)."""
        problems = []
        if backend == "yolo" and self.active() is None:
            problems.append("no active model")
        if backend == "template" and not self.labelled_images():
            problems.append("no labelled images")
        if self.master is None or self.master.layout is None:
            problems.append("no master")
        return problems


class PartRepository:
    """All part numbers under ``storage.parts_dir``."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self._parts: dict[str, Part] = {}
        self._lock = threading.RLock()

    def codes(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / PART_FILE).is_file())

    def __contains__(self, code: str) -> bool:
        return bool(code) and (self.root / code / PART_FILE).is_file()

    def get(self, code: str) -> Part | None:
        with self._lock:
            if code not in self:
                self._parts.pop(code, None)
                return None
            if code not in self._parts:
                self._parts[code] = Part(self.root, code)
            return self._parts[code]

    def create(self, code: str, description: str = "") -> Part:
        code = check_code(code)
        with self._lock:
            if code in self:
                raise ValueError(f"Part number {code} exists already")
            part = Part(self.root, code)
            part.description = description.strip()
            part.created = datetime.now().isoformat(timespec="seconds")
            part.save()
            self._parts[code] = part
            return part

    def delete(self, code: str) -> None:
        with self._lock:
            self._parts.pop(code, None)
            shutil.rmtree(self.root / code, ignore_errors=True)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
