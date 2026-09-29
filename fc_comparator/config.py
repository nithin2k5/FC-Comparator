"""Typed application configuration backed by a single YAML file.

Every setting (camera, detector, thresholds, alert backend, clip types, part
numbers, ...) lives in the YAML file. Relative paths are resolved against the
config file's folder.
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

from .models import PartNumber, Taxonomy

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
    scale: float = 0.5  # images are matched at this scale for speed
    max_per_class: int = 10  # templates per clip class (taken from marked boxes)
    min_score: float = 0.55  # normalized correlation needed to report a clip
    temperature: float = 0.04  # softmax temperature for class confidence


@dataclass
class DetectorConfig:
    backend: str = "auto"  # auto = YOLO if model_path exists, else template | yolo | template
    model_path: str = "models/clip_detector.pt"
    min_score: float = 0.25  # detections below this are ignored entirely
    confidence_threshold: float = 0.6  # detections below this are "uncertain" (= NG)
    imgsz: int = 960
    device: str = "cpu"
    template: TemplateDetectorConfig = field(default_factory=TemplateDetectorConfig)


@dataclass
class LayoutConfig:
    match_tolerance: float = 0.6  # max centre distance, as a fraction of the clip size
    max_shift_px: float = 200.0  # largest board shift accepted relative to the master image
    max_rotation_deg: float = 8.0
    min_matched_fraction: float = 0.5  # fewer matches => "board does not match this part's layout"


@dataclass
class AnnotationConfig:
    dir: str = "data/annotations"


@dataclass
class TrainingConfig:
    base_model: str = "yolo11n.pt"
    epochs: int = 60
    imgsz: int = 960
    batch: int = 8
    val_split: float = 0.2
    mirror: bool = True  # add mirrored copies (swapping mirror classes such as fork_left/right)
    patience: int = 20
    workdir: str = "runs/detector"


@dataclass
class CompareConfig:
    mode: str = "master"  # master (A) | cross (B) | both


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
class SecurityConfig:
    # PBKDF2 hashes, see fc_comparator.security. Empty = nobody can log in / acknowledge.
    setup_password: str = ""
    supervisor_pin: str = ""
    lock_on_ng: bool = True


@dataclass
class StorageConfig:
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
    appearance: str = "light"  # light | dark | system
    scale: float = 1.0


@dataclass
class AppConfig:
    station: StationConfig = field(default_factory=StationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    taxonomy: Taxonomy = field(default_factory=Taxonomy)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    annotations: AnnotationConfig = field(default_factory=AnnotationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    parts: dict[str, PartNumber] = field(default_factory=dict)

    # Not serialized: folder used to resolve relative paths and the file loaded from.
    base_dir: Path = field(default_factory=Path.cwd, repr=False, compare=False)
    path: Path | None = field(default=None, repr=False, compare=False)

    def resolve(self, p: str | os.PathLike) -> Path:
        """Resolve a config path relative to the config file's folder."""
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)

    def validate(self) -> list[str]:
        """Return a list of problems (empty when the config is usable)."""
        problems = list(self.taxonomy.validate())
        for pn in self.parts.values():
            try:
                pn.validate(self.taxonomy)
            except ValueError as exc:
                problems.append(str(exc))
        if self.compare.mode not in ("master", "cross", "both"):
            problems.append(f"compare.mode must be master, cross or both (got {self.compare.mode!r})")
        d = self.detector
        if not 0.0 <= d.min_score <= d.confidence_threshold <= 1.0:
            problems.append("need 0 <= detector.min_score <= detector.confidence_threshold <= 1")
        if d.backend not in ("auto", "yolo", "template"):
            problems.append(f"detector.backend must be auto, yolo or template (got {d.backend!r})")
        return problems

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in ("base_dir", "path"):
                continue
            value = getattr(self, f.name)
            if f.name == "parts":
                out[f.name] = {code: pn.to_dict() for code, pn in value.items()}
            elif f.name == "taxonomy":
                out[f.name] = value.to_dict()
            elif dataclasses.is_dataclass(value):
                out[f.name] = dataclasses.asdict(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "AppConfig":
        data = dict(data or {})
        parts = {str(code): PartNumber.from_dict(code, spec) for code, spec in (data.pop("parts", None) or {}).items()}
        taxonomy = Taxonomy.from_dict(data.pop("taxonomy", None))
        cfg = _build(cls, data)
        cfg.parts = parts
        cfg.taxonomy = taxonomy
        if base_dir is not None:
            cfg.base_dir = Path(base_dir)
        return cfg


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
