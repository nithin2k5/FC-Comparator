"""Typed station configuration backed by a single YAML file.

The station's settings (camera, detection, alerts, login, PIN, storage ...) live
in the YAML file. Part numbers - their images, labels, models and master - live
in their own folders under ``storage.parts_dir``. Relative paths are resolved
against the config file's folder.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class StationConfig:
    name: str = "Station-1"


@dataclass
class CameraConfig:
    source: str = "camera"  # camera | file
    index: int = 0
    backend: str = "auto"  # auto | dshow | msmf | v4l2 | gstreamer
    width: int = 1920
    height: int = 1080
    fps: int = 30
    autofocus: bool = False
    focus: float | None = None
    exposure: float | None = None
    warmup_frames: int = 5
    file_path: str = ""


@dataclass
class TemplateDetectorConfig:
    scale: float = 0.35  # images are matched at this scale for speed
    max_per_class: int = 8  # templates per label (taken from labelled boxes)
    max_negatives: int = 48  # "not an object" patches sampled/mined automatically from the labelled images
    min_score: float = 0.7  # normalized correlation needed to report an object
    temperature: float = 0.04  # softmax temperature for class confidence


@dataclass
class DetectorConfig:
    # yolo = each part's active trained model; template = template matching on the part's
    # labelled images (no training needed - for trials and tests)
    backend: str = "yolo"
    min_score: float = 0.25  # detections below this are ignored entirely
    confidence_threshold: float = 0.6  # detections below this are "uncertain" (= NG)
    imgsz: int = 960
    device: str = "cpu"
    # a trained model can only be put into use when its accuracy on the held-back images reaches this
    min_model_accuracy: float = 0.95
    template: TemplateDetectorConfig = field(default_factory=TemplateDetectorConfig)


@dataclass
class LayoutConfig:
    match_tolerance: float = 1.0  # max centre distance in object sizes (capped at 45% of the closest position spacing)
    max_shift_px: float = 200.0  # largest board shift accepted relative to the master image
    max_rotation_deg: float = 8.0
    min_matched_fraction: float = 0.5  # fewer matches => "board does not match this part's layout"


@dataclass
class TrainingConfig:
    base_model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 960
    batch: int = 8
    val_split: float = 0.2
    mirror: bool = True  # add mirrored copies when labels contain left/right (swapped in the copy)
    patience: int = 30
    workdir: str = "runs/detector"
    min_images: int = 5  # labelled images needed before training
    min_per_label: int = 3  # boxes every label needs before training


@dataclass
class GpioAlertConfig:
    green_pin: int = 17
    red_pin: int = 27
    buzzer_pin: int = 22
    active_high: bool = True


@dataclass
class UsbRelayAlertConfig:
    port: str = "COM3"
    baudrate: int = 9600
    green_channel: int = 1
    red_channel: int = 2
    buzzer_channel: int = 3


@dataclass
class ModbusAlertConfig:
    host: str = "192.168.0.10"
    port: int = 502
    unit_id: int = 1
    green_coil: int = 0
    red_coil: int = 1
    buzzer_coil: int = 2
    timeout: float = 0.5


@dataclass
class AlertConfig:
    backend: str = "console"  # console | gpio | usb_relay | modbus | none
    buzzer_mode: str = "pulse"  # pulse | until_ack
    buzzer_seconds: float = 2.0
    ok_light_seconds: float = 0.0  # 0 = keep green on until next inspection
    gpio: GpioAlertConfig = field(default_factory=GpioAlertConfig)
    usb_relay: UsbRelayAlertConfig = field(default_factory=UsbRelayAlertConfig)
    modbus: ModbusAlertConfig = field(default_factory=ModbusAlertConfig)


@dataclass
class AuthConfig:
    """The model login: needed for Model Setup and for the commands that change a model."""

    user: str = "nice"
    # PBKDF2 hash, see station.security. Empty = nobody can log in. Default password: nice1234
    password: str = (
        "pbkdf2_sha256$200000$018ae44a90f8104c8d1feee1df463e2c$e313d84171de1c68a126161c0b6b599aa36a20b6a3de29fabff9ba1e8887ef9b")
    setup_timeout_min: float = 10.0  # Model Setup closes after this long without input


@dataclass
class SecurityConfig:
    supervisor_pin: str = ""  # PBKDF2 hash; releases the NG lock. Empty = nobody can release it
    lock_on_ng: bool = True


@dataclass
class StorageConfig:
    parts_dir: str = "data/parts"  # one folder per part number: images, labels, models, master
    database: str = "data/inspections.db"
    image_dir: str = "data/images"
    save_ok_every_n: int = 20  # 0 = never save OK images
    save_raw_images: bool = False


@dataclass
class BarcodeConfig:
    # Regex applied to the scanned text; group "part" is the part number.
    part_pattern: str = r"^(?P<part>[A-Za-z0-9\-_]+)$"
    operator_prefix: str = "OP:"


@dataclass
class TriggerConfig:
    key: str = "F9"
    gpio_pin: int | None = None


@dataclass
class UiConfig:
    fullscreen: bool = False
    preview_fps: int = 15


@dataclass
class AppConfig:
    station: StationConfig = field(default_factory=StationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    ui: UiConfig = field(default_factory=UiConfig)

    # Not serialized: folder used to resolve relative paths and the file loaded from.
    base_dir: Path = field(default_factory=Path.cwd, repr=False, compare=False)
    path: Path | None = field(default=None, repr=False, compare=False)

    def resolve(self, p: str | os.PathLike) -> Path:
        """Resolve a config path relative to the config file's folder."""
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)

    def validate(self) -> list[str]:
        """Return a list of problems (empty when the config is usable)."""
        problems = []
        d = self.detector
        if not 0.0 <= d.min_score <= d.confidence_threshold <= 1.0:
            problems.append("need 0 <= detector.min_score <= detector.confidence_threshold <= 1")
        if not 0.0 <= d.min_model_accuracy <= 1.0:
            problems.append("detector.min_model_accuracy must be between 0 and 1")
        if d.backend not in ("yolo", "template"):
            problems.append(f"detector.backend must be yolo or template (got {d.backend!r})")
        if self.auth.setup_timeout_min <= 0:
            problems.append("auth.setup_timeout_min must be greater than 0")
        if self.training.min_images < 2 or self.training.min_per_label < 1:
            problems.append("training.min_images must be at least 2 and training.min_per_label at least 1")
        return problems

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in ("base_dir", "path"):
                continue
            value = getattr(self, f.name)
            out[f.name] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "AppConfig":
        cfg = _build(cls, _migrate(dict(data or {})))
        if base_dir is not None:
            cfg.base_dir = Path(base_dir)
        return cfg


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Accept config files written before part numbers got their own folders."""
    for key in ("compare", "annotations", "dataset", "taxonomy"):
        data.pop(key, None)
    if data.pop("parts", None):
        log.warning("config: 'parts' is no longer read - part numbers live in storage.parts_dir; "
                    "set their masters up again in Model Setup")
    sec = dict(data.get("security") or {})
    auth = dict(data.get("auth") or {})
    if "login_user" in sec:
        auth.setdefault("user", sec.pop("login_user"))
    if "login_password" in sec:
        auth.setdefault("password", sec.pop("login_password"))
    if "security" in data:
        data["security"] = sec
    if auth:
        data["auth"] = auth
    if "detector" in data:
        det = dict(data["detector"] or {})
        det.pop("model_path", None)
        if det.get("backend") == "auto":
            det["backend"] = "yolo"
        data["detector"] = det
    return data


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring (and logging) unknown keys."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    names = {f.name for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        if key not in names:
            log.warning("Unknown config key %s.%s ignored", cls.__name__, key)
            continue
        hint = hints[key]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _build(hint, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike) -> AppConfig:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = AppConfig.from_dict(data, base_dir=path.parent)
    cfg.path = path
    for problem in cfg.validate():
        log.warning("Config: %s", problem)
    return cfg


def save_config(cfg: AppConfig, path: str | os.PathLike | None = None) -> Path:
    """Atomically write the config (temp file + rename) so a crash never leaves it half written."""
    target = Path(path) if path else cfg.path
    if target is None:
        raise ValueError("No path given and config was not loaded from a file")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=None, width=120)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".yaml", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    cfg.path = target
    cfg.base_dir = target.parent
    return target
