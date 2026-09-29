"""Frame sources: a live OpenCV camera and an image file (for testing / offline use)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from ..config import AppConfig, CameraConfig

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CaptureError(RuntimeError):
    pass


class FrameSource(ABC):
    """Something that yields BGR frames."""

    is_live: bool = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def latest(self) -> np.ndarray | None:
        """Most recent frame (for preview). May return None before the first frame."""

    @abstractmethod
    def capture(self, timeout: float = 2.0) -> np.ndarray:
        """A frame taken *after* this call started (for inspection). Raises CaptureError."""

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread that also works with non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise CaptureError(f"Cannot read image {path}: {exc}") from exc
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise CaptureError(f"Cannot read image {path}")
    return img


def write_image(path: str | Path, image: np.ndarray, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise CaptureError(f"Cannot encode image {path}")
    buf.tofile(str(path))
    return path


class FileSource(FrameSource):
    """Serves a single image file. ``load()`` switches to another file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._image: np.ndarray | None = None

    def open(self) -> None:
        if self.path is not None and self._image is None:
            self.load(self.path)

    def load(self, path: str | Path) -> np.ndarray:
        self._image = read_image(path)
        self.path = Path(path)
        return self._image

    def set_image(self, image: np.ndarray) -> None:
        self._image = image

    def latest(self) -> np.ndarray | None:
        return None if self._image is None else self._image.copy()

    def capture(self, timeout: float = 2.0) -> np.ndarray:
        if self._image is None:
            raise CaptureError("No image loaded")
        return self._image.copy()


_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
}


class CameraSource(FrameSource):
    """OpenCV VideoCapture with a background grab thread.

    The thread keeps reading so the driver buffer never holds stale frames;
    ``capture()`` waits for a frame newer than the call, guaranteeing the image
    reflects the board as it is *now*. Reconnects automatically on failure.
    """

    is_live = True

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str = ""

    # -- lifecycle -------------------------------------------------------
    def open(self) -> None:
        if self._running:
            return
        self._open_device()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-grab", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_device(self) -> None:
        c = self.cfg
        backend = c.backend
        if backend == "auto" and sys.platform == "win32":
            backend = "dshow"  # fastest to open and honours resolution requests on Windows
        cap = cv2.VideoCapture(int(c.index), _BACKENDS.get(backend, cv2.CAP_ANY))
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"Cannot open camera index {c.index} (backend {backend})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c.autofocus else 0)
        if not c.autofocus and c.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, float(c.focus))
        if c.exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if backend == "v4l2" else 0)  # manual mode
            cap.set(cv2.CAP_PROP_EXPOSURE, float(c.exposure))
        for _ in range(max(0, c.warmup_frames)):
            cap.read()
        log.info(
            "Camera %s opened at %dx%d",
            c.index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._cap = cap
        self.error = ""

    def _loop(self) -> None:
        failures = 0
        while self._running:
            cap = self._cap
            ok, frame = (cap.read() if cap is not None else (False, None))
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._lock.notify_all()
                continue
            failures += 1
            if failures >= 10:
                self.error = "Camera disconnected - reconnecting"
                log.warning(self.error)
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                try:
                    self._open_device()
                    failures = 0
                except CaptureError as exc:
                    self.error = str(exc)
                    time.sleep(1.0)
            else:
                time.sleep(0.02)

    # -- frames ----------------------------------------------------------
    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def capture(self, timeout: float = 2.0) -> np.ndarray:
        start = time.monotonic()
        with self._lock:
            ok = self._lock.wait_for(lambda: self._frame is not None and self._frame_time > start, timeout)
            if not ok:
                raise CaptureError(self.error or "Timed out waiting for a camera frame")
            return self._frame.copy()


def create_source(cfg: AppConfig) -> FrameSource:
    cam = cfg.camera
    if cam.source == "file":
        return FileSource(cfg.resolve(cam.file_path) if cam.file_path else None)
    if cam.source == "camera":
        return CameraSource(cam)
    raise ValueError(f"Unknown camera.source {cam.source!r} (camera | file)")
