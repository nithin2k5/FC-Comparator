import numpy as np
import pytest

from fc_comparator.classify.dataset import harvest_board
from fc_comparator.synthetic import layout_rois, render_board

TEACH_LABELS = ["round", "fork_left", "small", "missing"]  # fork_right is learned by mirroring


def random_columns(rng: np.random.Generator, labels=TEACH_LABELS, cables=4, rows=4) -> list[list[str]]:
    return [[str(rng.choice(labels)) for _ in range(rows)] for _ in range(cables)]


def build_dataset(root, boards: int = 8, seed: int = 100):
    """Teach a dataset the way an operator would: known boards -> labeled crops."""
    rng = np.random.default_rng(seed)
    rois = layout_rois()
    for b in range(boards):
        cols = random_columns(rng)
        img = render_board(cols, seed=seed + b)
        for roi in rois:
            label = cols[roi.cable - 1][roi.row - 1]
            # harvest_board labels from a per-row pattern; here each cable differs, so call per position
            harvest_board(img, [roi], [label] * 4, root, fork_orientation=None, stem=f"b{b}", only={roi.key})
    return root


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("dataset"))
