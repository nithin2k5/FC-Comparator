import pytest

from fc_comparator.core.compare import compare
from fc_comparator.core.models import Classification, PartNumber


def rows_to_pattern(columns: list[list[str]]) -> list[list[str]]:
    """columns[cable-1][row-1] -> pattern[row-1][cable-1]"""
    return [list(r) for r in zip(*columns)]


P002 = PartNumber("P002", [["fork"] * 4, ["fork"] * 4, ["small"] * 4, ["fork"] * 4])
P001 = PartNumber("P001", [["round"] * 4, ["fork"] * 4, ["small"] * 4, ["round"] * 4])


def board(columns: list[list[str]], conf: float = 0.95) -> dict:
    """columns[cable-1][row-1] -> label"""
    return {(c + 1, r + 1): Classification(lbl, conf) for c, col in enumerate(columns) for r, lbl in enumerate(col)}


def bad(results):
    return sorted((p.cable, p.row) for p in results if not p.ok)


def test_master_all_ok():
    found = board([["fork_left", "fork_right", "small", "fork"]] * 4)
    res = compare(found, P002)
    assert len(res) == 16 and all(p.ok for p in res)
    assert [(p.cable, p.row) for p in res][:5] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]


def test_master_flags_every_mismatch_with_details():
    found = board([
        ["fork", "fork", "small", "fork"],
        ["round", "fork", "small", "fork"],   # wrong part laid down at row 1
        ["fork", "fork", "small", "small"],
        ["fork", "fork", "fork", "fork"],
    ])
    res = compare(found, P002)
    assert bad(res) == [(2, 1), (3, 4), (4, 3)]
    r = next(p for p in res if (p.cable, p.row) == (2, 1))
    assert (r.expected, r.found, r.confidence) == ("fork", "round", 0.95)
    assert r.reason == "round != fork"


def test_each_position_has_its_own_expected_clip():
    """Real boards carry different clips on different cables in the same row."""
    columns = [
        ["round", "small", "fork_right", "round"],
        ["round", "small", "metal_fork_right", "round"],
        ["round", "small", "fork_right", "round"],
        ["round", "small", "grey_fork_left", "round"],
    ]
    part = PartNumber("FCC-01", rows_to_pattern(columns))
    assert part.expected(4, 3) == "grey_fork_left" and (part.cables, part.rows) == (4, 4)
    assert all(p.ok for p in compare(board(columns), part))
    swapped = [list(c) for c in columns]
    swapped[1][2], swapped[3][2] = "grey_fork_left", "metal_fork_right"  # two forks swapped between cables
    res = compare(board(swapped), part)
    assert bad(res) == [(2, 3), (4, 3)]


def test_missing_and_uncertain_never_ok():
    found = board([["round", "fork", "small", "round"]] * 4)
    found[(1, 2)] = Classification("missing", 0.99)
    found[(3, 3)] = Classification("uncertain", 0.2)
    del found[(4, 4)]  # no classification at all -> uncertain
    res = compare(found, P001)
    assert bad(res) == [(1, 2), (3, 3), (4, 4)]
    reasons = {p.key: p.reason for p in res if not p.ok}
    assert reasons[(1, 2)] == "clip missing"
    assert reasons[(3, 3)] == "low confidence"


def test_fork_orientation_checked_when_specified():
    part = PartNumber("PX", rows_to_pattern([["fork_left", "round"], ["fork_left", "round"], ["grey_fork_right", "round"]]))
    found = board([
        ["fork_left", "round"],
        ["fork_right", "round"],
        ["grey_fork_left", "round"],
    ])
    res = compare(found, part)
    assert bad(res) == [(2, 1), (3, 1)]
    assert "orientation" in next(p for p in res if p.key == (2, 1)).reason
    assert "orientation" in next(p for p in res if p.key == (3, 1)).reason


def test_group_accepts_either_orientation_but_not_another_colour():
    part = PartNumber("PG", [["grey_fork", "fork"]])
    assert all(p.ok for p in compare(board([["grey_fork_left"], ["fork_right"]]), part))
    assert bad(compare(board([["fork_left"], ["metal_fork_right"]]), part)) == [(1, 1), (2, 1)]


def test_invalid_part():
    with pytest.raises(ValueError, match="one clip per cable"):
        compare({}, PartNumber("PB", [["round", "round"], ["round"]]))
    with pytest.raises(ValueError, match="invalid pattern labels"):
        compare({}, PartNumber("PB", [["round", "banana"]]))


def test_configurable_layout():
    part = PartNumber("P6", [[lbl] * 5 for lbl in ["round", "small", "fork", "round", "small", "fork"]])
    found = board([["round", "small", "fork_left", "round", "small", "fork_right"]] * 5)
    res = compare(found, part)
    assert len(res) == 30 and all(p.ok for p in res)
