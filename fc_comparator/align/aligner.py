"""Align a new board image to the reference image (ORB + RANSAC homography).

Features are computed on a downscaled grayscale copy for speed; the homography
is scaled back to full resolution and the new image is warped into the
reference frame so the taught ROIs line up again. The transform is sanity
checked (shift, rotation, scale) so a bad match can never silently move the
ROIs somewhere wrong.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from ..config import AlignmentConfig
from ..models import AlignmentInfo

log = logging.getLogger(__name__)


class Aligner:
    def __init__(self, cfg: AlignmentConfig, reference: np.ndarray | None = None):
        self.cfg = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.max_features, fastThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_shape: tuple[int, int] | None = None
        self._ref_kp = None
        self._ref_des = None
        self._scale = 1.0
        if reference is not None:
            self.set_reference(reference)

    @property
    def has_reference(self) -> bool:
        return self._ref_des is not None

    def set_reference(self, reference: np.ndarray) -> None:
        self._ref_shape = reference.shape[:2]
        self._scale = min(1.0, self.cfg.work_width / reference.shape[1])
        kp, des = self._features(reference)
        if des is None or len(kp) < self.cfg.min_inliers:
            log.warning("Reference image has only %d features - alignment will be unreliable", len(kp))
        self._ref_kp, self._ref_des = kp, des

    def _features(self, image: np.ndarray):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._scale != 1.0:
            gray = cv2.resize(gray, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        return self._orb.detectAndCompute(gray, None)

    def align(self, image: np.ndarray) -> tuple[np.ndarray, AlignmentInfo]:
        """Return (image warped into the reference frame, info).

        On failure the original image is returned with ``info.ok = False``.
        """
        if not self.has_reference:
            return image, AlignmentInfo(applied=False, ok=False, message="no reference image")
        if image.shape[:2] != self._ref_shape:
            return image, AlignmentInfo(
                applied=False, ok=False, message=f"image size {image.shape[1]}x{image.shape[0]} != reference"
            )

        kp, des = self._features(image)
        if des is None or len(kp) < 4:
            return image, AlignmentInfo(ok=False, message="no features in image")

        pairs = self._matcher.knnMatch(des, self._ref_des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.cfg.ratio_test * p[1].distance]
        if len(good) < self.cfg.min_inliers:
            return image, AlignmentInfo(ok=False, inliers=len(good), message=f"only {len(good)} feature matches")

        src = np.float32([kp[m.queryIdx].pt for m in good]) / self._scale
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in good]) / self._scale
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_threshold / self._scale)
        inliers = int(mask.sum()) if mask is not None else 0
        if h is None or inliers < self.cfg.min_inliers:
            return image, AlignmentInfo(ok=False, inliers=inliers, message=f"only {inliers} RANSAC inliers")

        problem, shift = self._check(h)
        if problem:
            return image, AlignmentInfo(ok=False, inliers=inliers, shift_px=shift, message=problem)

        rh, rw = self._ref_shape
        warped = cv2.warpPerspective(
            image, h, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return warped, AlignmentInfo(applied=True, ok=True, inliers=inliers, shift_px=shift, message="aligned")

    def _check(self, h: np.ndarray) -> tuple[str, float]:
        """Reject transforms that are not a small rigid-ish board movement."""
        rh, rw = self._ref_shape
        center = np.float32([[[rw / 2, rh / 2]]])
        moved = cv2.perspectiveTransform(center, h)[0, 0]
        shift = float(np.hypot(*(moved - center[0, 0])))
        a = h[:2, :2] / h[2, 2]
        scale = math.sqrt(abs(np.linalg.det(a)))
        rotation = math.degrees(math.atan2(a[1, 0], a[0, 0]))
        perspective = float(np.abs(h[2, :2] / h[2, 2]).max())
        c = self.cfg
        if shift > c.max_shift_px:
            return f"shift {shift:.0f}px exceeds {c.max_shift_px:.0f}px", shift
        if abs(rotation) > c.max_rotation_deg:
            return f"rotation {rotation:.1f} deg exceeds {c.max_rotation_deg} deg", shift
        if abs(scale - 1) > c.max_scale_change:
            return f"scale {scale:.2f} out of range", shift
        if perspective > 1e-3:
            return "excessive perspective distortion", shift
        return "", shift
