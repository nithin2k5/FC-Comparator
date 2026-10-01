import cv2
import numpy as np
import pytest

from fc_comparator.core.models import Box
from fc_comparator.vision.camera import write_image
from fc_comparator.vision.dataset import AnnotationStore
from fc_comparator.vision.synthetic import render_marked_board


@pytest.fixture
def board(tmp_path):
    img, boxes = render_marked_board([["round", "fork_left", "small", "round"]] * 4, seed=1)
    return write_image(tmp_path / "upload" / "board.jpg", img), img, boxes


def test_add_mark_and_reload(tmp_path, board):
    path, img, boxes = board
    store = AnnotationStore(tmp_path / "ann")
    image_id, added = store.add_image(path, part="P001", good=True)
    assert added and len(store) == 1
    rec = store.get(image_id)
    assert (rec.width, rec.height) == (1280, 960) and rec.source_name == "board.jpg" and rec.good
    assert store.image_path(image_id).read_bytes() == path.read_bytes()  # stored unmodified

    store.set_boxes(image_id, boxes)
    again = AnnotationStore(tmp_path / "ann")  # persisted to disk
    rec2 = again.get(image_id)
    assert len(rec2.boxes) == 16 and rec2.marked and rec2.part == "P001"
    assert rec2.boxes[0].label == boxes[0].label
    assert again.stats() == {"images": 1, "marked": 1, "good": 1, "boxes": 16,
                             "per_class": {"fork_left": 4, "round": 8, "small": 4}}
    assert np.array_equal(again.load_image(image_id), cv2.imread(str(path)))


def test_duplicate_upload_is_skipped(tmp_path, board):
    path = board[0]
    store = AnnotationStore(tmp_path / "ann")
    a, added_a = store.add_image(path)
    b, added_b = store.add_image(path)
    assert added_a and not added_b and a == b and len(store) == 1


def test_add_array_update_delete(tmp_path):
    store = AnnotationStore(tmp_path / "ann")
    img = np.full((100, 200, 3), 128, np.uint8)
    image_id, _ = store.add_image(img, name="camera")
    store.update(image_id, part="P009", good=True, note="first")
    rec = store.get(image_id)
    assert (rec.part, rec.good, rec.note, rec.width, rec.height) == ("P009", True, "first", 200, 100)
    # boxes are clipped to the image and degenerate boxes dropped
    store.set_boxes(image_id, [Box("round", -10, 90, 50, 30), Box("small", 5, 5, 1, 1)])
    assert [(b.x, b.y, b.w, b.h) for b in store.get(image_id).boxes] == [(0.0, 90.0, 40.0, 10.0)]
    f = store.image_path(image_id)
    store.delete(image_id)
    assert len(store) == 0 and not f.exists()


def test_labels_add_rename_merge_delete(tmp_path, board):
    store = AnnotationStore(tmp_path / "ann")
    image_id, _ = store.add_image(board[0])
    store.set_boxes(image_id, board[2])
    assert store.labels == ["round", "fork_left", "small"]  # labels used by boxes are known
    assert store.add_label(" tape ") == "tape" and store.labels[-1] == "tape"
    with pytest.raises(ValueError):
        store.add_label("two words")
    assert store.rename_label("small", "fir_tree") == 4
    by = store.boxes_by_class()
    assert set(by) == {"round", "fork_left", "fir_tree"} and len(by["round"]) == 8
    assert store.labels == ["round", "fork_left", "fir_tree", "tape"]
    assert store.rename_label("fir_tree", "round") == 4  # an existing name: merged
    assert store.label_counts() == {"round": 12, "fork_left": 4, "tape": 0}
    assert store.delete_label("fork_left") == 4
    again = AnnotationStore(tmp_path / "ann")  # persisted
    assert again.labels == ["round", "tape"] and len(again.get(image_id).boxes) == 12


def test_rejects_non_images(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        AnnotationStore(tmp_path / "ann").add_image(p)
