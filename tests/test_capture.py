import numpy as np
import pytest

from fc_comparator.capture import CaptureError, FileSource, create_source
from fc_comparator.capture.sources import read_image, write_image
from fc_comparator.config import AppConfig
from fc_comparator.synthetic import render_board


def test_file_source_roundtrip(tmp_path):
    img = render_board([["round", "fork_left", "small", "missing"]] * 4, seed=1)
    path = write_image(tmp_path / "sub dir" / "böard.png", img)  # spaces + non-ASCII
    src = FileSource(path)
    with src:
        a = src.capture()
        b = src.latest()
    assert np.array_equal(a, img) and np.array_equal(b, img)
    a[0, 0] = 0  # returned copies must not alias the source image
    assert not np.array_equal(a, src.capture())


def test_file_source_errors(tmp_path):
    with pytest.raises(CaptureError):
        FileSource().capture()
    with pytest.raises(CaptureError):
        read_image(tmp_path / "nope.png")


def test_create_source_from_config(tmp_path):
    cfg = AppConfig(base_dir=tmp_path)
    cfg.camera.source = "file"
    cfg.camera.file_path = "x.png"
    src = create_source(cfg)
    assert isinstance(src, FileSource) and src.path == tmp_path / "x.png"
    cfg.camera.source = "bogus"
    with pytest.raises(ValueError):
        create_source(cfg)


def test_synthetic_board_is_deterministic():
    cols = [["round", "fork_right", "small", "round"]] * 4
    assert np.array_equal(render_board(cols, seed=3), render_board(cols, seed=3))
    assert not np.array_equal(render_board(cols, seed=3), render_board(cols, seed=4))
