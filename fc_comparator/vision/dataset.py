"""Annotation store: uploaded board images and the clip boxes marked on them.

Layout on disk::

    <root>/index.json          metadata + boxes for every image
    <root>/images/<id>.jpg     the uploaded images (copied in, never modified)

One JSON index keeps the whole data set consistent (it is written atomically),
is easy to back up, and is exported to the YOLO format only when training.
Duplicate uploads (same file content) are detected and skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..core.models import Box
from .camera import IMAGE_EXTENSIONS, read_image, write_image

INDEX = "index.json"


@dataclass
class ImageRecord:
    id: str
    file: str  # relative to the store root
    width: int
    height: int
    sha1: str
    source_name: str = ""
    part: str = ""  # part number shown on the board (optional)
    good: bool = False  # a known-good board (can become a part's master)
    added: str = ""
    boxes: list[Box] = field(default_factory=list)
    note: str = ""

    @property
    def marked(self) -> bool:
        return bool(self.boxes)

    def to_dict(self) -> dict:
        return {
            "file": self.file, "width": self.width, "height": self.height, "sha1": self.sha1,
            "source_name": self.source_name, "part": self.part, "good": self.good, "added": self.added,
            "note": self.note, "boxes": [b.to_dict() for b in self.boxes],
        }

    @classmethod
    def from_dict(cls, image_id: str, d: dict) -> "ImageRecord":
        return cls(
            id=image_id, file=d["file"], width=int(d["width"]), height=int(d["height"]), sha1=d.get("sha1", ""),
            source_name=d.get("source_name", ""), part=d.get("part", ""), good=bool(d.get("good", False)),
            added=d.get("added", ""), note=d.get("note", ""), boxes=[Box.from_dict(b) for b in d.get("boxes", [])],
        )


class AnnotationStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._records: dict[str, ImageRecord] = {}
        self.reload()

    # -- persistence -----------------------------------------------------------
    def reload(self) -> None:
        with self._lock:
            path = self.root / INDEX
            self._records = {}
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                for image_id, d in data.get("images", {}).items():
                    self._records[image_id] = ImageRecord.from_dict(image_id, d)

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "images": {i: r.to_dict() for i, r in self._records.items()}}
        fd, tmp = tempfile.mkstemp(prefix=".index-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            os.replace(tmp, self.root / INDEX)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- queries ---------------------------------------------------------------
    def records(self) -> list[ImageRecord]:
        """All images in upload order (kept in the index file, so results are reproducible)."""
        with self._lock:
            return list(self._records.values())

    def get(self, image_id: str) -> ImageRecord:
        with self._lock:
            return self._records[image_id]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._records

    def image_path(self, image_id: str) -> Path:
        return self.root / self.get(image_id).file

    def load_image(self, image_id: str) -> np.ndarray:
        return read_image(self.image_path(image_id))

    def stats(self) -> dict:
        recs = self.records()
        counts = Counter(b.label for r in recs for b in r.boxes)
        return {
            "images": len(recs),
            "marked": sum(r.marked for r in recs),
            "good": sum(r.good for r in recs),
            "boxes": sum(counts.values()),
            "per_class": dict(sorted(counts.items())),
        }

    def boxes_by_class(self) -> dict[str, list[tuple[str, Box]]]:
        """label -> [(image_id, box), ...] over all marked images."""
        out: dict[str, list[tuple[str, Box]]] = {}
        for r in self.records():
            for b in r.boxes:
                out.setdefault(b.label, []).append((r.id, b))
        return out

    # -- changes ---------------------------------------------------------------
    def add_image(
        self, source: str | os.PathLike | np.ndarray, *, part: str = "", good: bool = False, name: str = ""
    ) -> tuple[str, bool]:
        """Add an image (a file path or a BGR array). Returns (image_id, added).

        ``added`` is False when identical content is already in the store; the
        existing id is returned instead.
        """
        if isinstance(source, np.ndarray):
            img = source
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("Cannot encode image")
            data = buf.tobytes()
            name = name or "capture"
            ext = ".jpg"
        else:
            path = Path(source)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Not an image file: {path.name}")
            data = path.read_bytes()
            img = read_image(path)
            name = name or path.name
            ext = path.suffix.lower() if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") else ".png"
        sha1 = hashlib.sha1(data).hexdigest()
        with self._lock:
            for r in self._records.values():
                if r.sha1 == sha1:
                    return r.id, False
            image_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            rel = f"images/{image_id}{ext}"
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, np.ndarray) or Path(source).suffix.lower() == ext:
                target.write_bytes(data)  # keep the original bytes (no re-compression)
            else:
                write_image(target, img)  # e.g. .tif -> .png
            h, w = img.shape[:2]
            self._records[image_id] = ImageRecord(
                image_id, rel, w, h, sha1, source_name=name, part=part.strip(), good=good,
                added=datetime.now().isoformat(timespec="seconds"),
            )
            self._save()
            return image_id, True

    def set_boxes(self, image_id: str, boxes: list[Box]) -> None:
        with self._lock:
            rec = self._records[image_id]
            rec.boxes = [self._clip(b, rec) for b in boxes if b.w >= 2 and b.h >= 2]
            self._save()

    def update(self, image_id: str, *, part: str | None = None, good: bool | None = None, note: str | None = None) -> None:
        with self._lock:
            rec = self._records[image_id]
            if part is not None:
                rec.part = part.strip()
            if good is not None:
                rec.good = bool(good)
            if note is not None:
                rec.note = note
            self._save()

    def delete(self, image_id: str) -> None:
        with self._lock:
            rec = self._records.pop(image_id)
            self._save()
            (self.root / rec.file).unlink(missing_ok=True)

    def rename_label(self, old: str, new: str) -> int:
        """Rename a clip class in every box (after a taxonomy change). Returns boxes changed."""
        n = 0
        with self._lock:
            for rec in self._records.values():
                for b in rec.boxes:
                    if b.label == old:
                        b.label = new
                        n += 1
            if n:
                self._save()
        return n

    @staticmethod
    def _clip(b: Box, rec: ImageRecord) -> Box:
        x0, y0 = max(0.0, b.x), max(0.0, b.y)
        x1, y1 = min(float(rec.width), b.x + b.w), min(float(rec.height), b.y + b.h)
        return Box(b.label, x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), 1.0)
