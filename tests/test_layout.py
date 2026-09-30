import numpy as np
import pytest

from fc_comparator.config import LayoutConfig
from fc_comparator.core.layout import (
    LayoutError,
    infer_grid,
    master_from_boxes,
    match_layout,
)
from fc_comparator.core.models import Box, Taxonomy
from fc_comparator.vision.synthetic import Geometry, render_marked_board

P001 = ["round", "fork_left", "small", "round"]
TAX = Taxonomy()


def boxes_for(columns, **kw):
    return render_marked_board(columns, noise=0, **kw)[1]


def test_infer_grid_orders_cables_and_rows():
    boxes = boxes_for([P001] * 4, angle=2.0, shift=(30, -20))
    grid = infer_grid(boxes, cables=4, rows=4)
    assert (grid.cables, grid.rows) == (4, 4)
    for (c, r), idx in grid.cells.items():
        assert len(idx) == 1
        assert boxes[idx[0]].label == P001[r - 1]
    xs = [grid.centers[(c, 1)][0] for c in range(1, 5)]
    assert xs == sorted(xs)


def test_infer_grid_handles_other_layouts():
    geo = Geometry(cable_x0=200, cable_dx=180, row_y0=260, row_dy=120)
    cols = [["small", "round", "round", "small", "fork_right"]] * 6
    grid = infer_grid(boxes_for(cols, geometry=geo))
    assert (grid.cables, grid.rows) == (6, 5)


def test_infer_grid_wrong_counts():
    with pytest.raises(LayoutError, match="cable column"):
        infer_grid(boxes_for([P001] * 3), cables=4)
    with pytest.raises(LayoutError, match="no clips"):
        infer_grid([])


def test_master_from_good_board():
    img_boxes = boxes_for([P001] * 4)
    pattern, layout = master_from_boxes(img_boxes, (1280, 960))
    assert pattern == [[lbl] * 4 for lbl in P001]
    assert (layout.cables, layout.rows) == (4, 4)
    assert layout.positions[(3, 1)] == pytest.approx((780, 300), abs=6)  # round clip, centred on the cable
    assert all(type(v) is float for xy in layout.positions.values() for v in xy)  # YAML-serializable
    assert 55 < layout.clip_size[0] < 80


def test_master_keeps_a_different_clip_per_cable():
    cols = [list(P001) for _ in range(4)]
    cols[0][0] = "small"
    cols[3][1] = "fork_right"
    pattern, _ = master_from_boxes(boxes_for(cols), (1280, 960))
    assert pattern[0] == ["small", "round", "round", "round"]
    assert pattern[1] == ["fork_left", "fork_left", "fork_left", "fork_right"]


def test_master_rejects_incomplete_boards():
    cols = [list(P001) for _ in range(4)]
    cols[1][2] = "missing"
    with pytest.raises(LayoutError, match="cable 2 row 3 has no clip"):
        master_from_boxes(boxes_for(cols), (1280, 960))


@pytest.fixture(scope="module")
def layout():
    return master_from_boxes(boxes_for([P001] * 4, seed=1), (1280, 960))[1]


@pytest.mark.parametrize("shift,angle", [((0, 0), 0), ((20, 0), 0), ((-20, 20), 0), ((60, -45), 2.5), ((-35, 10), -3)])
def test_match_layout_shifted_boards(layout, shift, angle):
    dets = boxes_for([P001] * 4, shift=shift, angle=angle, seed=7)
    pl = match_layout(dets, layout, LayoutConfig(), (1280, 960))
    assert pl.info.ok and pl.info.matched == 16 and pl.extras == []
    for (c, r), j in pl.assignment.items():
        assert dets[j].label == P001[r - 1]
    assert pl.info.shift_px == pytest.approx(np.hypot(*shift), abs=12)


def test_match_layout_missing_and_extra(layout):
    cols = [list(P001) for _ in range(4)]
    cols[1][2] = "missing"
    cols[3][0] = "missing"
    dets = boxes_for(cols, shift=(15, 10), seed=8)
    dets.append(Box("small", 640, 520, 36, 56))  # a clip where the master has none
    pl = match_layout(dets, layout, LayoutConfig(), (1280, 960))
    missing = sorted(k for k, j in pl.assignment.items() if j is None)
    assert missing == [(2, 3), (4, 1)]
    assert [dets[j].label for j in pl.extras] == ["small"]
    assert pl.info.ok and pl.info.matched == 14
    ex, ey = pl.expected[(2, 3)]
    assert (ex, ey) == pytest.approx((520 + 15, 640 + 10), abs=8)  # where the missing clip should be


def test_match_layout_whole_row_missing(layout):
    cols = [[P001[0], P001[1], "missing", P001[3]] for _ in range(4)]
    pl = match_layout(boxes_for(cols, seed=9), layout, LayoutConfig(), (1280, 960))
    assert sorted(k for k, j in pl.assignment.items() if j is None) == [(c, 3) for c in range(1, 5)]


def test_match_layout_wrong_board(layout):
    geo = Geometry(cable_x0=180, cable_dx=130, row_y0=250, row_dy=95)  # a different part's layout
    dets = boxes_for([["round"] * 7] * 8, geometry=geo)
    pl = match_layout(dets, layout, LayoutConfig(max_shift_px=60), (1280, 960))
    assert not pl.info.ok or pl.extras  # never a clean match


def test_match_layout_no_detections(layout):
    pl = match_layout([], layout, LayoutConfig(), (1280, 960))
    assert not pl.info.ok and all(j is None for j in pl.assignment.values())


def test_match_layout_scales_with_camera_resolution(layout):
    dets = [Box(b.label, b.x * 1.5, b.y * 1.5, b.w * 1.5, b.h * 1.5) for b in boxes_for([P001] * 4, seed=3)]
    pl = match_layout(dets, layout, LayoutConfig(), (1920, 1440))
    assert pl.info.ok and pl.info.matched == 16
