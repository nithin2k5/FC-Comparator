"""Part folders: images, labels, model versions and master per part number."""

import json

import pytest

from fc_comparator.core.models import PartNumber
from fc_comparator.vision.parts import PartRepository, check_code
from fc_comparator.vision.synthetic import render_marked_board


def fake_model(part, version, accuracy, pr=None):
    part.models_dir.mkdir(parents=True, exist_ok=True)
    (part.models_dir / f"{version}.pt").write_bytes(b"weights")
    (part.models_dir / f"{version}.json").write_text(json.dumps(
        {"accuracy": accuracy, "trained": "2026-10-01T10:00:00", "labelled_images": 12,
         "precision_recall": pr or {}}), encoding="utf-8")


def test_create_reload_and_delete(tmp_path):
    repo = PartRepository(tmp_path / "parts")
    assert repo.codes() == [] and repo.get("P1") is None
    part = repo.create("P1", " Harness one ")
    assert part.description == "Harness one" and part.created
    with pytest.raises(ValueError, match="exists"):
        repo.create("P1")
    for bad in ("", "a b", "../x", "x/y"):
        with pytest.raises(ValueError):
            check_code(bad)
    img, boxes = render_marked_board([["round", "fork_left"]] * 2, seed=1)
    image_id, _ = part.store.add_image(img)
    part.store.set_boxes(image_id, boxes)
    assert (part.dir / "images").is_dir() and part.labels == ["round", "fork_left"]

    again = PartRepository(tmp_path / "parts").get("P1")  # from disk
    assert again.description == "Harness one" and again.labelled_images() == 1
    assert again.taxonomy().classes == ["round", "fork_left", "fork_right"]
    repo.delete("P1")
    assert repo.codes() == [] and not part.dir.exists()


def test_model_versions_activation_and_weak_labels(tmp_path):
    part = PartRepository(tmp_path).create("P1")
    assert part.next_version() == "v001" and part.active() is None
    fake_model(part, "v001", 0.97)
    fake_model(part, "v002", 0.80, {"round": [0.99, 0.98], "small": [0.9, 0.5]})
    assert [m.version for m in part.models()] == ["v001", "v002"] and part.next_version() == "v003"
    v2 = part.model("v002")
    assert not v2.meets(0.95) and part.model("v001").meets(0.95)
    assert v2.weak_labels(0.95) == [("small", 0.9, 0.5)]
    assert "v002 (80.0%, trained 2026-10-01 10:00, 12 images)" == v2.summary()
    part.set_active("v001")
    assert PartRepository(tmp_path).get("P1").active().version == "v001"
    with pytest.raises(KeyError):
        part.set_active("v009")


def test_setup_problems_and_master(tmp_path):
    part = PartRepository(tmp_path).create("P1", "desc")
    assert part.setup_problems("yolo") == ["no active model", "no master"]
    assert part.setup_problems("template") == ["no labelled images", "no master"]
    fake_model(part, "v001", 0.99)
    part.set_active("v001")
    from fc_comparator.core.layout import master_from_boxes

    _img, boxes = render_marked_board([["round", "small"]] * 3, seed=2)
    pattern, layout = master_from_boxes(boxes, (1280, 960))
    part.set_master(PartNumber("P1", pattern, layout=layout, master_image="img"))
    assert part.setup_problems("yolo") == []
    again = PartRepository(tmp_path).get("P1")
    assert again.master.pattern == [["round"] * 3, ["small"] * 3] and again.master.description == "desc"


def test_label_rename_reaches_the_master_and_delete_is_guarded(tmp_path):
    part = PartRepository(tmp_path).create("P1")
    img, boxes = render_marked_board([["round", "small"]] * 2, seed=3)
    image_id, _ = part.store.add_image(img)
    part.store.set_boxes(image_id, boxes)
    part.set_master(PartNumber("P1", [["round", "round"], ["small", "small"]]))
    assert part.rename_label("small", "tiny") == 2
    assert part.master.pattern[1] == ["tiny", "tiny"] and "tiny" in part.labels
    with pytest.raises(ValueError, match="used by the master"):
        part.delete_label("tiny")
    part.store.add_label("tape")
    part.delete_label("tape")
    assert "tape" not in part.labels
