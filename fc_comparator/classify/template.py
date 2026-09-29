"""OpenCV template-matching classifier (fallback before a YOLO model is trained).

Each class has many templates (labeled crops from the dataset folder). A crop
is resized slightly larger than the templates so ``matchTemplate`` can absorb
a few pixels of residual misalignment. A class score is the mean of its best
few template scores (robust to one lucky/unlucky template), and confidence is
a softmax over class scores: when two classes score similarly - the
prototype's fork-vs-round confusion - confidence drops and the position
becomes "uncertain" instead of a confident wrong answer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import TemplateConfig
from ..models import CLASSIFIER_LABELS, Classification, mirror_label
from .base import Classifier
from .dataset import list_images

log = logging.getLogger(__name__)

TOP_K = 3


class TemplateClassifier(Classifier):
    name = "template"

    def __init__(self, cfg: TemplateConfig, dataset_dir: Path | None = None):
        self.cfg = cfg
        self.templates: dict[str, list[np.ndarray]] = {}
        self._cache_key = None
        self._cache: tuple[np.ndarray, list[str]] | None = None
        if dataset_dir is not None:
            self.load(dataset_dir)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.templates.items() if v)

    # -- templates ---------------------------------------------------------
    def load(self, dataset_dir: Path) -> None:
        """(Re)load templates: up to ``max_templates_per_class`` per class, evenly sampled."""
        from ..capture.sources import read_image

        self.templates = {}
        dataset_dir = Path(dataset_dir)
        for label in CLASSIFIER_LABELS:
            files = list_images(dataset_dir / label)
            n = self.cfg.max_templates_per_class
            if len(files) > n:
                idx = np.linspace(0, len(files) - 1, n).round().astype(int)
                files = [files[i] for i in idx]
            for f in files:
                try:
                    self.add(label, read_image(f), mirror=False)
                except Exception as exc:  # corrupt file must not stop the station
                    log.warning("Skipping template %s: %s", f, exc)
        if self.cfg.mirror_forks:
            originals = {k: list(self.templates.get(k, [])) for k in ("fork_left", "fork_right")}
            for label, temps in originals.items():
                self.templates.setdefault(mirror_label(label), []).extend(cv2.flip(t, 1) for t in temps)
        log.info("Template classifier: %s", {k: len(v) for k, v in self.templates.items()})

    def add(self, label: str, crop: np.ndarray, mirror: bool | None = None) -> None:
        t = self._prep(crop, self.cfg.size)
        self.templates.setdefault(label, []).append(t)
        if (self.cfg.mirror_forks if mirror is None else mirror) and mirror_label(label) != label:
            self.templates.setdefault(mirror_label(label), []).append(cv2.flip(t, 1))

    @staticmethod
    def _prep(img: np.ndarray, size: int) -> np.ndarray:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        interp = cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_LINEAR
        return cv2.resize(gray, (size, size), interpolation=interp)

    # -- classification ---------------------------------------------------
    def _template_matrix(self) -> tuple[np.ndarray, list[str]]:
        """All templates as zero-mean unit-norm rows (cached until templates change)."""
        key = tuple((k, len(v)) for k, v in self.templates.items())
        if self._cache_key != key:
            rows, owners = [], []
            for label, temps in self.templates.items():
                for t in temps:
                    rows.append(t.astype(np.float32).ravel())
                    owners.append(label)
            mat = np.stack(rows) if rows else np.zeros((0, self.cfg.size**2), np.float32)
            self._cache = (_normalize_rows(mat), owners)
            self._cache_key = key
        return self._cache

    def class_scores_batch(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """Per-crop class scores = mean of the best TOP_K normalized correlations.

        Equivalent to ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)`` for every
        (crop, template, offset), but done as one matrix product: every
        search-window position of every crop against every template.
        """
        tmat, owners = self._template_matrix()
        if not owners or not crops:
            return [{} for _ in crops]
        s, m = self.cfg.size, self.cfg.search_margin
        probes = np.stack([self._prep(c, s + 2 * m).astype(np.float32) for c in crops])
        windows = np.lib.stride_tricks.sliding_window_view(probes, (s, s), axis=(1, 2))
        n_off = windows.shape[1] * windows.shape[2]
        # Templates are zero-mean, so <window - mean, t> == <window, t>: the numerator needs no
        # per-window normalization. The window norms come from integral images.
        num = windows.reshape(len(crops) * n_off, s * s) @ tmat.T
        denom = np.concatenate([_window_norms(p, s).ravel() for p in probes])
        corr = (num / denom[:, None]).reshape(len(crops), n_off, len(owners)).max(axis=1)  # crops x templates
        labels = list(dict.fromkeys(owners))
        owner_idx = np.array([labels.index(o) for o in owners])
        out = []
        for row in corr:
            scores = {}
            for li, label in enumerate(labels):
                vals = np.sort(row[owner_idx == li])[::-1]
                scores[label] = float(vals[:TOP_K].mean())
            out.append(scores)
        return out

    def class_scores(self, crop: np.ndarray) -> dict[str, float]:
        return self.class_scores_batch([crop])[0]

    def classify(self, crops: list[np.ndarray]) -> list[Classification]:
        return [self._decide(s) for s in self.class_scores_batch(crops)]

    def _decide(self, scores: dict[str, float]) -> Classification:
        if not scores:
            return Classification("uncertain", 0.0, {})
        labels = list(scores)
        raw = np.array([scores[k] for k in labels], np.float64)
        e = np.exp((raw - raw.max()) / max(self.cfg.temperature, 1e-6))
        probs = e / e.sum()
        best = int(np.argmax(probs))
        conf = float(probs[best])
        if raw[best] < self.cfg.min_match_score:
            # Nothing looks like this crop: never be confident about it.
            conf = min(conf, max(0.0, float(raw[best])))
        return Classification(labels[best], conf, {k: float(p) for k, p in zip(labels, probs)})


def _window_norms(img: np.ndarray, s: int) -> np.ndarray:
    """||w - mean(w)|| for every s x s window of ``img`` (valid positions), via integral images."""
    ii, ii2 = cv2.integral2(img.astype(np.float64))
    total = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
    total2 = ii2[s:, s:] - ii2[:-s, s:] - ii2[s:, :-s] + ii2[:-s, :-s]
    norm = np.sqrt(np.maximum(total2 - total * total / (s * s), 0.0))
    # A (near-)uniform window correlates with nothing: infinite norm -> score 0.
    return np.where(norm < 1.0 * s, np.inf, norm).astype(np.float32)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat - mat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norm, 1e-6)
