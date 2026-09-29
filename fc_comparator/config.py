"""Typed application configuration backed by a single YAML file.

Every setting (camera, thresholds, ROIs, alert backend, part numbers, ...) lives
in the YAML file. Relative paths are resolved against the config file's folder.
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

from .models import PartNumber, Roi

log = logging.getLogger(__name__)


@dataclass
class StationConfig:
    name: str = "Station-1"
    cables: int = 4
    rows: int = 4


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
class AlignmentConfig:
    enabled: bool = True
    max_features: int = 3000
    work_width: int = 1000  # images are downscaled to this width for feature matching
    ratio_test: float = 0.75
    min_inliers: int = 25
    ransac_threshold: float = 4.0
    max_shift_px: float = 80.0
    max_rotation_deg: float = 10.0
    max_scale_change: float = 0.1
    fail_as_ng: bool = True


@dataclass
class TemplateConfig:
    size: int = 64
    search_margin: int = 6
    max_templates_per_class: int = 40
    temperature: float = 0.04
    min_match_score: float = 0.4
    mirror_forks: bool = True


@dataclass
class YoloConfig:
    imgsz: int = 128
    device: str = "cpu"


@dataclass
class ClassifierConfig:
    backend: str = "auto"  # auto | yolo | template
    model_path: str = "models/clip_classifier.pt"
    dataset_dir: str = "data/dataset"
    confidence_threshold: float = 0.6
    roi_padding: float = 0.0  # fraction of ROI size added on every side before cropping
    template: TemplateConfig = field(default_factory=TemplateConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)


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
    # PBKDF2 hashes, see fc_comparator.security. Defaults: setup "admin", supervisor PIN "1234".
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
    font_scale: float = 1.0


@dataclass
class AppConfig:
    station: StationConfig = field(default_factory=StationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    reference_image: str = "data/reference.png"
    rois: list[Roi] = field(default_factory=list)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
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

    # ---- helpers -------------------------------------------------------
    def resolve(self, p: str | os.PathLike) -> Path:
        """Resolve a config path relative to the config file's folder."""
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)

    def roi_map(self) -> dict[tuple[int, int], Roi]:
        return {r.key: r for r in self.rois}

    def validate(self) -> list[str]:
        """Return a list of problems (empty when the config is usable)."""
        problems: list[str] = []
        s = self.station
        if s.cables < 1 or s.rows < 1:
            problems.append("station.cables and station.rows must be >= 1")
        seen: set[tuple[int, int]] = set()
        for r in self.rois:
            if not (1 <= r.cable <= s.cables and 1 <= r.row <= s.rows):
                problems.append(f"ROI cable {r.cable} row {r.row} is outside the {s.cables}x{s.rows} layout")
            if r.w <= 0 or r.h <= 0:
                problems.append(f"ROI cable {r.cable} row {r.row} has non-positive size")
            if r.key in seen:
                problems.append(f"Duplicate ROI for cable {r.cable} row {r.row}")
            seen.add(r.key)
        missing = [(c, r) for c in range(1, s.cables + 1) for r in range(1, s.rows + 1) if (c, r) not in seen]
        if missing:
            problems.append(f"{len(missing)} positions have no ROI: {missing[:6]}{'...' if len(missing) > 6 else ''}")
        for pn in self.parts.values():
            try:
                pn.validate(s.rows)
            except ValueError as exc:
                problems.append(str(exc))
        if self.compare.mode not in ("master", "cross", "both"):
            problems.append(f"compare.mode must be master, cross or both (got {self.compare.mode!r})")
        if not 0.0 <= self.classifier.confidence_threshold <= 1.0:
            problems.append("classifier.confidence_threshold must be between 0 and 1")
        return problems

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in ("base_dir", "path"):
                continue
            value = getattr(self, f.name)
            if f.name == "rois":
                out[f.name] = [r.to_dict() for r in value]
            elif f.name == "parts":
                out[f.name] = {code: pn.to_dict() for code, pn in value.items()}
            elif dataclasses.is_dataclass(value):
                out[f.name] = dataclasses.asdict(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "AppConfig":
        data = dict(data or {})
        rois = [Roi(**{k: int(v) for k, v in r.items()}) for r in data.pop("rois", None) or []]
        parts_raw = data.pop("parts", None) or {}
        parts = {}
        for code, spec in parts_raw.items():
            spec = spec or {}
            if isinstance(spec, list):  # shorthand: P001: [round, fork, ...]
                spec = {"pattern": spec}
            parts[str(code)] = PartNumber(
                code=str(code), pattern=[str(p) for p in spec.get("pattern", [])], description=str(spec.get("description", ""))
            )
        cfg = _build(cls, data)
        cfg.rois = rois
        cfg.parts = parts
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
