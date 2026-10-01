"""Generate the synthetic boards the tests use (tests/fixtures).

    python scripts/make_test_fixtures.py

Outputs
  tests/fixtures/parts/<code>/    parts P001-P003: labelled images (the part's master + mixed boards)
                                  and the master (pattern + layout) made from the master image
  tests/fixtures/boards/*.jpg     test boards (not used for learning)
  tests/fixtures/manifest.yaml    part number and expected result per test board
  tests/fixtures/config.yaml      the station settings the tests use (template matching, no trained model)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fc_comparator.config import load_config, save_config  # noqa: E402
from fc_comparator.core.parts import part_from_boxes  # noqa: E402
from fc_comparator.vision.camera import write_image  # noqa: E402
from fc_comparator.vision.parts import PartRepository  # noqa: E402
from fc_comparator.vision.synthetic import (  # noqa: E402
    DEFAULT_GEOMETRY,
    Geometry,
    render_marked_board,
)

P001 = ["round", "fork_left", "small", "round"]
P002 = ["fork_right", "fork_right", "small", "fork_right"]
P003 = ["small", "round", "fork_left", "round", "small"]
GEO3 = Geometry(cable_x0=330, cable_dx=310, row_y0=260, row_dy=135)  # a part with another layout
PARTS = {
    "P001": ("Harness A", P001, 4, DEFAULT_GEOMETRY),
    "P002": ("Harness B", P002, 4, DEFAULT_GEOMETRY),
    "P003": ("Harness C", P003, 3, GEO3),
}


def cols(pattern, cables=4, changes=()):
    c = [list(pattern) for _ in range(cables)]
    for cable, row, label in changes:
        c[cable - 1][row - 1] = label
    return c


# name, part, columns, geometry, render kwargs, expected NG [(cable,row)], extras, error expected, note
BOARDS = [
    ("board_P001_ok", "P001", cols(P001), None, {}, [], 0, False, "all correct"),
    ("board_P002_ok", "P002", cols(P002), None, {}, [], 0, False, "all correct"),
    ("board_P003_ok", "P003", cols(P003, 3), GEO3, {}, [], 0, False, "all correct (other layout)"),
    # P001's master has left-facing forks, P002's right-facing: row 2 differs by orientation too
    ("board_P001_mixed", "P001", [P001, P001, P002, P001], None, {}, [(3, 1), (3, 2), (3, 4)], 0, False,
     "cable 3 is a P002 harness"),
    ("board_P002_mixed", "P002", [P001, P002, P002, P002], None, {}, [(1, 1), (1, 2), (1, 4)], 0, False,
     "cable 1 is a P001 harness"),
    ("board_P001_missing", "P001", cols(P001, changes=[(2, 3, "missing")]), None, {}, [(2, 3)], 0, False,
     "clip missing on cable 2 row 3"),
    ("board_P001_shifted_ok", "P001", cols(P001), None, {"shift": (20, -15), "angle": 1.0}, [], 0, False,
     "board shifted 20/-15 px, 1 deg"),
    ("board_P001_shifted_mixed", "P001", [P001, P002, P001, P001], None, {"shift": (-18, 20)},
     [(2, 1), (2, 2), (2, 4)], 0, False, "shifted -18/20 px, cable 2 is P002"),
    ("board_P002_wrong_clip", "P002", cols(P002, changes=[(4, 2, "round"), (3, 3, "fork_right")]), None, {},
     [(4, 2), (3, 3)], 0, False, "round instead of fork at C4R2, fork instead of small at C3R3"),
    ("board_P003_missing", "P003", cols(P003, 3, [(2, 4, "missing")]), GEO3, {"shift": (12, 18)}, [(2, 4)], 0, False,
     "other layout, shifted, clip missing on cable 2 row 4"),
    ("board_P003_orientation", "P003", cols(P003, 3, [(1, 3, "fork_right")]), GEO3, {}, [(1, 3)], 0, False,
     "fork faces the wrong way on cable 1 row 3"),
    ("board_P001_extra_clip", "P001", cols(P001), None, {"extra_clips": [("small", 650, 555)]}, [], 1, False,
     "an extra clip between the cables"),
    ("board_P003_as_P001", "P001", cols(P003, 3), GEO3, {}, None, 0, True, "a P003 board inspected as P001"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures")
    ap.add_argument("--config", type=Path, default=ROOT / "tests" / "fixtures" / "config.yaml")
    args = ap.parse_args()
    out: Path = args.out
    for sub in ("boards", "annotations", "dataset", "parts"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for old in ("reference.jpg", "manifest.yaml"):
        (out / old).unlink(missing_ok=True)

    # 1. mixed boards every part learns from (all labels, both layouts)
    rng = np.random.default_rng(7)
    labels = ["round", "fork_left", "fork_right", "small", "missing"]
    mixed = []
    for i in range(6):
        geo = GEO3 if i == 5 else DEFAULT_GEOMETRY
        n_cables, n_rows = (3, 5) if geo is GEO3 else (4, 4)
        mix = [[str(rng.choice(labels)) for _ in range(n_rows)] for _ in range(n_cables)]
        mixed.append((f"mixed_{i + 1}.jpg", *render_marked_board(mix, seed=40 + i, geometry=geo,
                                                                 shift=tuple(rng.uniform(-15, 15, 2)))))

    # 2. per part: the master board + the mixed boards, labelled; the master from the master image
    cfg = load_config(args.config)
    repo = PartRepository(cfg.resolve(cfg.storage.parts_dir))
    for i, (code, (desc, pattern, cables, geo)) in enumerate(PARTS.items()):
        part = repo.create(code, desc)
        img, boxes = render_marked_board(cols(pattern, cables), seed=10 + i, geometry=geo)
        master_id, _ = part.store.add_image(img, good=True, name=f"{code}_master.jpg")
        part.store.set_boxes(master_id, boxes)
        for name, mimg, mboxes in mixed:
            image_id, _ = part.store.add_image(mimg, name=name)
            part.store.set_boxes(image_id, mboxes)
        rec = part.store.get(master_id)
        part.set_master(part_from_boxes(code, rec.boxes, (rec.width, rec.height), part.taxonomy(), desc, rec.id))
    save_config(cfg)

    # 3. test boards + manifest
    manifest = []
    for i, (name, part, columns, geo, kw, expected, extras, error, note) in enumerate(BOARDS):
        img, _ = render_marked_board(columns, seed=100 + i, geometry=geo or DEFAULT_GEOMETRY, **kw)
        write_image(out / "boards" / f"{name}.jpg", img, quality=92)
        manifest.append({"image": f"boards/{name}.jpg", "part": part, "expected_ng": [list(p) for p in expected or []],
                         "extras": extras, "error": error, "note": note})
    # a gloved finger over a clip: must never be OK
    img, _ = render_marked_board(cols(P001), seed=150)
    cv2.ellipse(img, (770, 475), (70, 45), 20, 0, 360, (150, 175, 205), -1, cv2.LINE_AA)
    cv2.ellipse(img, (770, 475), (70, 45), 20, 0, 360, (110, 130, 160), 3, cv2.LINE_AA)
    write_image(out / "boards" / "board_P001_obstructed.jpg", img, quality=92)
    manifest.append({"image": "boards/board_P001_obstructed.jpg", "part": "P001", "expected_ng": [[3, 2]], "extras": 0,
                     "error": False, "note": "a finger covers the clip at C3R2"})
    (out / "manifest.yaml").write_text(
        "# Expected results for the sample boards (used by tests/test_e2e.py)\n"
        + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=None), encoding="utf-8")
    print(f"Wrote parts {repo.codes()} to {repo.root} and {len(manifest)} test boards")


if __name__ == "__main__":
    main()
