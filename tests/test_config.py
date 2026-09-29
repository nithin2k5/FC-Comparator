from pathlib import Path

import pytest

from fc_comparator import models as m
from fc_comparator.config import AppConfig, load_config, save_config
from fc_comparator.models import PartNumber, Roi
from fc_comparator.security import hash_secret, verify_secret

SAMPLE = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_sample_config_is_valid():
    cfg = load_config(SAMPLE)
    assert cfg.validate() == []
    assert cfg.station.cables == 4 and cfg.station.rows == 4
    assert len(cfg.rois) == 16
    assert set(cfg.parts) == {"P001", "P002"}
    assert cfg.parts["P002"].pattern == ["fork", "fork", "small", "fork"]
    assert verify_secret("admin", cfg.security.setup_password)
    assert verify_secret("1234", cfg.security.supervisor_pin)


def test_relative_paths_resolve_against_config_folder():
    cfg = load_config(SAMPLE)
    assert cfg.resolve(cfg.reference_image) == SAMPLE.parent / "../samples/reference.jpg"
    assert cfg.resolve(cfg.reference_image).is_file()


def test_roundtrip(tmp_path):
    cfg = load_config(SAMPLE)
    cfg.parts["P003"] = PartNumber("P003", ["small", "small", "round", "fork_left"], "new")
    cfg.alert.backend = "modbus"
    out = save_config(cfg, tmp_path / "c.yaml")
    again = load_config(out)
    assert again.to_dict() == cfg.to_dict()
    assert again.parts["P003"].pattern[-1] == "fork_left"


def test_validate_reports_problems():
    cfg = AppConfig()
    cfg.station.cables, cfg.station.rows = 2, 2
    cfg.rois = [Roi(1, 1, 0, 0, 10, 10), Roi(1, 1, 0, 0, 10, 10), Roi(3, 1, 0, 0, 10, 10)]
    cfg.parts = {"X": PartNumber("X", ["round", "missing"])}
    problems = "\n".join(cfg.validate())
    assert "Duplicate ROI" in problems
    assert "outside" in problems
    assert "no ROI" in problems
    assert "invalid pattern labels" in problems


def test_part_validation():
    PartNumber("P", ["round", "fork"]).validate(2)
    with pytest.raises(ValueError):
        PartNumber("P", ["round"]).validate(2)
    with pytest.raises(ValueError):
        PartNumber("P", ["round", "uncertain"]).validate(2)


def test_label_rules():
    assert m.label_matches("fork", "fork_left")
    assert m.label_matches("fork", "fork")
    assert m.label_matches("fork_left", "fork_left")
    assert not m.label_matches("fork_left", "fork_right")
    assert not m.label_matches("fork_left", "fork")  # orientation unknown is not OK
    assert not m.label_matches("round", "missing")
    assert not m.label_matches("round", "uncertain")
    assert not m.label_matches("missing", "missing")


def test_secret_hashing():
    h = hash_secret("s3cret", iterations=1000)
    assert verify_secret("s3cret", h)
    assert not verify_secret("wrong", h)
    assert not verify_secret("s3cret", "")
    assert not verify_secret("s3cret", "garbage")
