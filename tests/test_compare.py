import pytest

from fc_comparator.compare import compare
from fc_comparator.models import Classification, PartNumber

P002 = PartNumber("P002", ["fork", "fork", "small", "fork"])
P001 = PartNumber("P001", ["round", "fork", "small", "round"])


def board(columns: list[list[str]], conf: float = 0.95) -> dict:
    """columns[cable-1][row-1] -> label"""
    return {(c + 1, r + 1): Classification(lbl, conf) for c, col in enumerate(columns) for r, lbl in enumerate(col)}


def bad(results):
    return sorted((p.cable, p.row) for p in results if not p.ok)


def test_master_all_ok():
    found = board([["fork_left", "fork_right", "small", "fork"]] * 4)
    res = compare(found, P002, 4, 4)
    assert len(res) == 16 and all(p.ok for p in res)
    assert [(p.cable, p.row) for p in res][:5] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]


def test_master_flags_every_mismatch_with_details():
    found = board([
        ["fork", "fork", "small", "fork"],
        ["round", "fork", "small", "fork"],   # wrong part laid down at row 1
        ["fork", "fork", "small", "small"],
        ["fork", "fork", "fork", "fork"],
    ])
    res = compare(found, P002, 4, 4)
    assert bad(res) == [(2, 1), (3, 4), (4, 3)]
    r = next(p for p in res if (p.cable, p.row) == (2, 1))
    assert (r.expected, r.found, r.confidence) == ("fork", "round", 0.95)
    assert r.reason == "round != fork"


def test_missing_and_uncertain_never_ok():
    found = board([["round", "fork", "small", "round"]] * 4)
    found[(1, 2)] = Classification("missing", 0.99)
    found[(3, 3)] = Classification("uncertain", 0.2)
    del found[(4, 4)]  # no classification at all -> uncertain
    res = compare(found, P001, 4, 4)
    assert bad(res) == [(1, 2), (3, 3), (4, 4)]
    reasons = {p.key: p.reason for p in res if not p.ok}
    assert reasons[(1, 2)] == "clip missing"
    assert reasons[(3, 3)] == "low confidence"


def test_fork_orientation_checked_when_specified():
    part = PartNumber("PX", ["fork_left", "round", "round", "round"])
    found = board([
        ["fork_left", "round", "round", "round"],
        ["fork_right", "round", "round", "round"],
        ["fork", "round", "round", "round"],  # classifier could not tell orientation
    ])
    res = compare(found, part, 3, 4)
    assert bad(res) == [(2, 1), (3, 1)]
    assert "orientation" in next(p for p in res if p.key == (2, 1)).reason


def test_cross_flags_odd_one_out():
    found = board([
        ["fork", "fork", "small", "fork"],
        ["fork", "fork", "small", "fork"],
        ["round", "fork", "small", "fork"],
        ["fork", "fork", "small", "missing"],
    ])
    res = compare(found, None, 4, 4, mode="cross")
    assert bad(res) == [(3, 1), (4, 4)]
    odd = next(p for p in res if p.key == (3, 1))
    assert odd.expected == "fork" and odd.found == "round" and "odd one out" in odd.reason


def test_cross_without_majority_flags_whole_row():
    found = board([
        ["round", "fork", "small", "fork"],
        ["round", "fork", "small", "fork"],
        ["fork", "fork", "small", "fork"],
        ["fork", "fork", "small", "fork"],
    ])
    res = compare(found, None, 4, 4, mode="cross")
    assert bad(res) == [(1, 1), (2, 1), (3, 1), (4, 1)]


def test_cross_majority_of_missing_is_not_ok():
    found = board([["missing", "fork"], ["missing", "fork"], ["missing", "fork"]])
    res = compare(found, None, 3, 2, mode="cross")
    assert bad(res) == [(1, 1), (2, 1), (3, 1)]


def test_cross_passes_uniform_wrong_board_but_both_catches_it():
    # All four cables are the wrong part: cross-check alone cannot see that.
    found = board([["round", "fork", "small", "round"]] * 4)
    assert all(p.ok for p in compare(found, None, 4, 4, mode="cross"))
    res = compare(found, P002, 4, 4, mode="both")
    assert bad(res) == [(c, r) for c in range(1, 5) for r in (1, 4)]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compare({}, P001, 4, 4, mode="nope")
    with pytest.raises(ValueError):
        compare({}, None, 4, 4, mode="master")
    with pytest.raises(ValueError):
        compare({}, P001, 4, 3)  # pattern length != rows


def test_configurable_layout():
    part = PartNumber("P6", ["round", "small", "fork", "round", "small", "fork"])
    found = board([part.pattern] * 5)
    res = compare(found, part, 5, 6)
    assert len(res) == 30 and all(p.ok for p in res)
