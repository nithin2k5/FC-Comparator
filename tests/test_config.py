from pathlib import Path

import pytest

from fc_comparator.config import AppConfig, load_config, save_config
from fc_comparator.models import Layout, PartNumber, Taxonomy
from fc_comparator.security import hash_secret, verify_secret

SAMPLE = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_sample_config_is_valid():
    cfg = load_config(SAMPLE)
    assert cfg.validate() == []
    assert set(cfg.parts) >= {"P001", "P002"}
    assert cfg.parts["P002"].cables == 4 and cfg.parts["P002"].rows == 4
    assert cfg.taxonomy.classes == ["round", "fork_left", "fork_right", "small"]
    assert verify_secret("5678", cfg.security.setup_password)
    assert verify_secret("1234", cfg.security.supervisor_pin)


def test_relative_paths_resolve_against_config_folder():
    cfg = load_config(SAMPLE)
    assert cfg.resolve(cfg.annotations.dir) == SAMPLE.parent / "../samples/annotations"


def test_roundtrip_with_layout(tmp_path):
    cfg = load_config(SAMPLE)
    layout = Layout(2, 3, {(c, r): (100.0 * c, 50.0 * r) for c in (1, 2) for r in (1, 2, 3)}, (60.0, 70.0), (1280, 960))
    cfg.parts["P003"] = PartNumber("P003", ["small", "fork", "round"], "new", cables=2, layout=layout, master_image="img1")
    cfg.taxonomy.classes.append("clamp")
    out = save_config(cfg, tmp_path / "c.yaml")
    again = load_config(out)
    assert again.to_dict() == cfg.to_dict()
    p3 = again.parts["P003"]
    assert p3.layout.positions[(2, 3)] == (200.0, 150.0) and p3.master_image == "img1"
    assert "clamp" in again.taxonomy.classes


def test_validate_reports_problems():
    cfg = AppConfig()
    cfg.parts = {"X": PartNumber("X", ["round", "missing"]), "Y": PartNumber("Y", ["round"], cables=0)}
    cfg.detector.min_score = 0.9
    problems = "\n".join(cfg.validate())
    assert "invalid pattern labels" in problems
    assert "cables must be >= 1" in problems
    assert "min_score" in problems


def test_part_validation():
    PartNumber("P", ["round", "fork"]).validate()
    with pytest.raises(ValueError):
        PartNumber("P", []).validate()
    with pytest.raises(ValueError):
        PartNumber("P", ["round", "uncertain"]).validate()
    layout = Layout(4, 3, {}, (1, 1), (1, 1))
    with pytest.raises(ValueError, match="layout"):
        PartNumber("P", ["round", "fork"], layout=layout).validate()


def test_taxonomy_rules():
    t = Taxonomy()
    assert t.matches("fork", "fork_left") and t.matches("fork", "fork_right")
    assert t.matches("fork_left", "fork_left")
    assert not t.matches("fork_left", "fork_right")
    assert not t.matches("round", "missing") and not t.matches("round", "uncertain")
    assert not t.matches("missing", "missing")
    assert t.group_of("fork_right") == "fork" and t.group_of("round") == "round"
    assert t.mirror_of("fork_left") == "fork_right" and t.mirror_of("small") == "small"
    assert t.expected_labels() == ["round", "fork_left", "fork_right", "small", "fork"]
    assert t.validate() == []


def test_taxonomy_validation():
    t = Taxonomy(classes=["round", "missing", "round", "bad name"], groups={"g": ["nope"]}, mirror={"round": "x"})
    problems = "\n".join(t.validate())
    for text in ("reserved", "duplicates", "invalid class name", "unknown classes", "mirror pair"):
        assert text in problems


def test_secret_hashing():
    h = hash_secret("s3cret", iterations=1000)
    assert verify_secret("s3cret", h)
    assert not verify_secret("wrong", h)
    assert not verify_secret("s3cret", "")
    assert not verify_secret("s3cret", "garbage")
