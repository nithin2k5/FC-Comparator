"""Template-matching clip detector (works from the first few marked images).

Templates
    The marked boxes in the annotation store: up to ``max_per_class`` per clip
    class, spread over different images, plus mirrored copies for mirror classes
    (a marked fork_left also yields fork_right templates).

Hard negatives
    Textured patches of the marked images that contain no clip (cable
    segments, connectors, board marks) are sampled automatically and compete as
    a "background" class. A location that looks more like bare cable than like
    any clip is therefore rejected - no fragile absolute threshold needed.

Detection
    The image is matched at ``scale`` against every template; per class the
    best response per location forms a class map. Peaks of the best clip map
    above ``min_score`` that beat the background are clips. Confidence is a
    softmax over all class maps (background included) at the peak, so a clip
    that resembles two classes gets a low confidence and is reported as
    "uncertain" instead of a confident wrong answer.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ...config import TemplateDetectorConfig
from ...core.models import Box, Taxonomy
from ..dataset import AnnotationStore
from .base import Detector

log = logging.getLogger(__name__)

BACKGROUND = "__background__"
MIN_TEMPLATE_PX = 6
MIN_PATCH_STD = 6.0  # a flat (featureless) patch is never a clip
NEGATIVE_MIN_STD = 18.0
NEGATIVE_MAX_OVERLAP = 0.15  # fraction of a negative patch allowed to overlap a marked clip


class TemplateDetector(Detector):
    name = "template"

    def __init__(self, cfg: TemplateDetectorConfig, taxonomy: Taxonomy, store: AnnotationStore | None = None):
        self.cfg = cfg
        self.taxonomy = taxonomy
        self.templates: dict[str, list[np.ndarray]] = {}
        self.negatives: list[np.ndarray] = []
        if store is not None:
            self.load(store)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.templates.items() if v)

    # -- templates -----------------------------------------------------------
    def load(self, store: AnnotationStore) -> None:
        s = self.cfg.scale
        cache: dict[str, np.ndarray | None] = {}

        def gray_of(image_id: str) -> np.ndarray | None:
            if image_id not in cache:
                try:
                    cache[image_id] = cv2.cvtColor(store.load_image(image_id), cv2.COLOR_BGR2GRAY)
                except Exception as exc:  # a broken file must not stop the station
                    log.warning("Skipping image %s: %s", image_id, exc)
                    cache[image_id] = None
            return cache[image_id]

        originals: dict[str, list[np.ndarray]] = {}
        for label, items in store.boxes_by_class().items():
            if label not in self.taxonomy.classes:
                continue
            n = self.cfg.max_per_class
            if len(items) > n:  # spread the choice over the whole data set
                items = [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]
            for image_id, b in items:
                gray = gray_of(image_id)
                t = None if gray is None else self._scaled_crop(gray, b.x, b.y, b.w, b.h)
                if t is not None:
                    originals.setdefault(label, []).append(t)
        templates: dict[str, list[np.ndarray]] = {k: list(v) for k, v in originals.items()}
        for label, temps in originals.items():
            twin = self.taxonomy.mirror_of(label)
            if twin != label and twin in self.taxonomy.classes:
                templates.setdefault(twin, []).extend(cv2.flip(t, 1) for t in temps)
        # Clip templates: drop only near-copies, natural variation must stay covered.
        self.templates = {k: _dedupe(v, threshold=0.985, size=24) for k, v in templates.items()}
        self.negatives = _dedupe(self._sample_negatives(store, gray_of))
        self._mine_hard_negatives(store, gray_of)
        log.info("Template detector: %s + %d negatives", {k: len(v) for k, v in self.templates.items()}, len(self.negatives))

    def _mine_hard_negatives(self, store: AnnotationStore, gray_of, rounds: int = 2, images: int = 8) -> None:
        """Run the detector on the marked images: whatever it finds that was not marked is not a clip.

        Relies on marked images being *completely* marked (every clip boxed).
        """
        marked = [r for r in store.records() if r.boxes][:images]
        for _ in range(rounds):
            budget = self.cfg.max_negatives - len(self.negatives)
            if budget <= 0:
                return
            found = []
            for rec in marked:
                gray = gray_of(rec.id)
                if gray is None:
                    continue
                for d in self.detect(gray):
                    if all(d.iou(b) < 0.3 for b in rec.boxes):
                        found.append((d.confidence, rec.id, d))
            if not found:
                return
            found.sort(key=lambda f: -f[0])
            mined = [self._scaled_crop(gray_of(i), d.x, d.y, d.w, d.h) for _c, i, d in found]
            self.negatives = _dedupe(self.negatives + [t for t in mined if t is not None])[: self.cfg.max_negatives]

    def _scaled_crop(self, gray: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray | None:
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        crop = gray[y0:y0 + int(round(h)), x0:x0 + int(round(w))]
        tw, th = int(round(crop.shape[1] * self.cfg.scale)), int(round(crop.shape[0] * self.cfg.scale))
        if tw < MIN_TEMPLATE_PX or th < MIN_TEMPLATE_PX:
            return None
        t = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        return t if t.std() >= MIN_PATCH_STD else None  # flat templates correlate with nothing (NaN)

    def _sample_negatives(self, store: AnnotationStore, gray_of) -> list[np.ndarray]:
        """Textured, clip-free patches of clip size from the marked images (a third of the budget;
        the rest is filled by hard-negative mining)."""
        rng = np.random.default_rng(0)
        marked = [r for r in store.records() if r.boxes]
        budget = self.cfg.max_negatives // 3
        if not marked or budget <= 0:
            return []
        per_image = max(1, int(np.ceil(budget / len(marked))))
        out: list[np.ndarray] = []
        for rec in marked:
            gray = gray_of(rec.id)
            if gray is None:
                continue
            w = float(np.median([b.w for b in rec.boxes]))
            h = float(np.median([b.h for b in rec.boxes]))
            H, W = gray.shape
            clip_mask = np.zeros((H, W), np.uint8)
            for b in rec.boxes:
                clip_mask[int(max(0, b.y)):int(b.y + b.h), int(max(0, b.x)):int(b.x + b.w)] = 1
            ii = cv2.integral(clip_mask)
            cands = []
            step = max(4, int(min(w, h) / 2))
            for y0 in range(0, int(H - h), step):
                for x0 in range(0, int(W - w), step):
                    y1, x1 = int(y0 + h), int(x0 + w)
                    overlap = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / max(1.0, w * h)
                    if overlap > NEGATIVE_MAX_OVERLAP:
                        continue
                    std = float(gray[y0:y1, x0:x1].std())
                    if std >= NEGATIVE_MIN_STD:
                        cands.append((x0, y0))
            if not cands:
                continue
            for i in rng.choice(len(cands), size=min(per_image, len(cands)), replace=False):
                x0, y0 = cands[int(i)]
                t = self._scaled_crop(gray, x0, y0, w, h)
                if t is not None:
                    out.append(t)
        return out[:budget]

    # -- detection -----------------------------------------------------------
    def detect(self, image: np.ndarray) -> list[Box]:
        if not self.templates:
            return []
        s = self.cfg.scale
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else gray
        H, W = small.shape
        labels = self.labels
        groups = [self.templates[lbl] for lbl in labels] + [self.negatives]
        maps = np.full((len(groups), H, W), -1.0, np.float32)
        for gi, temps in enumerate(groups):
            for t in temps:
                th, tw = t.shape
                if th > H or tw > W:
                    continue
                r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
                oy, ox = th // 2, tw // 2  # response index -> template centre
                view = maps[gi, oy:oy + r.shape[0], ox:ox + r.shape[1]]
                np.maximum(view, r, out=view)
        # box size per class: median template size (templates of one class are alike)
        class_size = [np.median([t.shape for t in self.templates[lbl]], axis=0) for lbl in labels]

        clip_maps = maps[: len(labels)]
        best = clip_maps.max(axis=0)
        best_cls = clip_maps.argmax(axis=0)
        med = float(np.median([min(t.shape) for temps in self.templates.values() for t in temps]))
        k = max(3, int(med * 0.5) | 1)
        # A negative patch rarely lines up exactly with a peak (e.g. the cable sits off-centre in
        # the patch), so let background evidence count within half a clip of the peak.
        background = cv2.dilate(maps[len(labels)], np.ones((k, k), np.uint8))
        maps[len(labels)] = background
        peaks = (best >= cv2.dilate(best, np.ones((k, k), np.uint8))) & (best >= self.cfg.min_score) & (best > background)
        ys, xs = np.nonzero(peaks)
        if len(ys) == 0:
            return []
        order = np.argsort(-best[ys, xs])
        ys, xs = ys[order], xs[order]

        # Greedy suppression: one clip per neighbourhood (vectorised distance check).
        kept_pts = np.empty((0, 2), np.float32)
        out = []
        for y, x in zip(ys, xs):
            li = int(best_cls[y, x])
            th, tw = (float(v) for v in class_size[li])
            if len(kept_pts):
                d = np.abs(kept_pts - (x, y))
                if np.any((d[:, 0] < tw * 0.6) & (d[:, 1] < th * 0.6)):
                    continue
            patch = small[max(0, int(y - th / 2)):int(y + th / 2), max(0, int(x - tw / 2)):int(x + tw / 2)]
            if patch.size == 0 or patch.std() < MIN_PATCH_STD:
                continue
            kept_pts = np.vstack([kept_pts, (x, y)])
            scores = maps[:, y, x].astype(np.float64)  # clip classes + background
            e = np.exp((scores - scores.max()) / max(self.cfg.temperature, 1e-6))
            conf = float(e[li] / e.sum())
            cx, cy, w, h = x / s, y / s, tw / s, th / s
            out.append(Box(labels[li], cx - w / 2, cy - h / 2, w, h, conf))
        return out


def _dedupe(templates: list[np.ndarray], threshold: float = 0.93, size: int = 16) -> list[np.ndarray]:
    """Drop templates nearly identical to one already kept: they cost time and add nothing."""
    kept, vecs = [], []
    for t in templates:
        v = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        v = v / n if n > 1e-6 else v
        if vecs and float(np.max(np.stack(vecs) @ v)) > threshold:
            continue
        kept.append(t)
        vecs.append(v)
    return kept
