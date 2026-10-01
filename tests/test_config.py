from pathlib import Path

import pytest

from fc_comparator.config import AppConfig, load_config, save_config
from fc_comparator.core.models import Layout, PartNumber, Taxonomy, check_label, mirror_name
from fc_comparator.station.security import hash_secret, verify_secret

ROOT = Path(__file__).resolve().parents[1]
STATION = ROOT / "config" / "config.yaml"
SAMPLE = ROOT / "tests" / "fixtures" / "config.yaml"


def test_station_config_is_valid():
    cfg = load_config(STATION)
    assert cfg.validate() == []
    assert cfg.resolve(cfg.storage.parts_dir).resolve() == ROOT / "data" / "parts"
    assert cfg.auth.user == "nice" and cfg.auth.setup_timeout_min == 10
    assert verify_secret("nice1234", cfg.auth.password)
    assert verify_secret("1234", cfg.security.supervisor_pin)
    assert cfg.detector.backend == "yolo" and cfg.detector.min_model_accuracy == pytest.approx(0.95)


def test_sample_config_is_valid():
    cfg = load_config(SAMPLE)
    assert cfg.validate() == []
    assert cfg.detector.backend == "template"
    assert cfg.resolve(cfg.storage.parts_dir) == SAMPLE.parent / "parts"


def test_older_config_files_still_load(tmp_path, caplog):
    """Before part numbers had their own folders: login under security, taxonomy, data set and parts in the file."""
    old = tmp_path / "old.yaml"
    old.write_text("annotations: {dir: marks}\ncompare: {mode: master}\ntaxonomy: {classes: [round]}\n"
                   "dataset: {dir: ds}\ndetector: {backend: auto, model_path: m.pt}\n"
                   "security: {login_user: boss, login_password: h, supervisor_pin: p, lock_on_ng: false}\n"
                   "parts:\n  P9: {pattern: [[round]]}\n", encoding="utf-8")
    cfg = load_config(old)
    assert (cfg.auth.user, cfg.auth.password) == ("boss", "h")
    assert (cfg.security.supervisor_pin, cfg.security.lock_on_ng) == ("p", False)
    assert cfg.detector.backend == "yolo"
    assert "parts" in caplog.text


def test_roundtrip(tmp_path):
    cfg = load_config(SAMPLE)
    cfg.auth.setup_timeout_min = 3
    cfg.training.min_images = 7
    out = save_config(cfg, tmp_path / "c.yaml")
    again = load_config(out)
    assert again.to_dict() == cfg.to_dict()
    assert again.auth.setup_timeout_min == 3 and again.training.min_images == 7


def test_validate_reports_problems():
    cfg = AppConfig()
    cfg.detector.min_score = 0.9
    cfg.detector.backend = "auto"
    cfg.auth.setup_timeout_min = 0
    problems = "\n".join(cfg.validate())
    for text in ("min_score", "backend", "setup_timeout_min"):
        assert text in problems


def test_part_validation():
    PartNumber("P", [["round", "fork"]]).validate()
    with pytest.raises(ValueError):
        PartNumber("P", []).validate()
    with pytest.raises(ValueError, match="one object per cable"):
        PartNumber("P", [["round", "fork"], ["round"]]).validate()
    with pytest.raises(ValueError, match="unknown labels"):
        PartNumber("P", [["round", "banana"]]).validate(Taxonomy.from_labels(["round"]))
    layout = Layout(4, 3, {}, (1, 1), (1, 1))
    with pytest.raises(ValueError, match="layout"):
        PartNumber("P", [["round", "fork"]], layout=layout).validate()


def test_part_master_roundtrip():
    layout = Layout(2, 3, {(c, r): (100.0 * c, 50.0 * r) for c in (1, 2) for r in (1, 2, 3)}, (60.0, 70.0), (1280, 960))
    pn = PartNumber("P3", [["small", "round"], ["fork", "fork"], ["round", "small"]], "d", layout, "img1")
    again = PartNumber.from_dict("P3", pn.to_dict(), "d")
    assert again == pn and again.layout.positions[(2, 3)] == (200.0, 150.0)


def test_taxonomy_from_labels_adds_mirror_twins():
    t = Taxonomy.from_labels(["round", "fork_left", "Left-clip", "bright", "tape"])
    assert t.classes == ["round", "fork_left", "Left-clip", "bright", "tape", "fork_right", "Right-clip"]
    assert t.mirror_of("fork_left") == "fork_right" and t.mirror_of("fork_right") == "fork_left"
    assert t.mirror_of("bright") == "bright" and t.mirror_of("round") == "round"
    assert t.matches("fork_left", "fork_left") and not t.matches("fork_left", "fork_right")
    assert not t.matches("round", "missing") and not t.matches("round", "uncertain")
    assert not t.matches("missing", "missing")
    assert mirror_name("left") == "right" and mirror_name("clip_right_side") == "clip_left_side"


def test_label_names():
    assert check_label(" red_clip ") == "red_clip"
    for bad in ("", "two words", "missing", "Uncertain", "_x", "a/b"):
        with pytest.raises(ValueError):
            check_label(bad)


def test_secret_hashing():
    h = hash_secret("s3cret", iterations=1000)
    assert verify_secret("s3cret", h)
    assert not verify_secret("wrong", h)
    assert not verify_secret("s3cret", "")
    assert not verify_secret("s3cret", "garbage")
