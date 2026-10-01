import numpy as np
import pytest

from fc_comparator.core.models import Taxonomy
from fc_comparator.vision.dataset import AnnotationStore
from fc_comparator.vision.synthetic import render_marked_board

CLIP_LABELS = ["round", "fork_left", "fork_right", "small", "missing"]
TAX = Taxonomy.from_labels(["round", "fork_left", "fork_right", "small"])


def random_columns(rng: np.random.Generator, labels=CLIP_LABELS, cables=4, rows=4) -> list[list[str]]:
    return [[str(rng.choice(labels)) for _ in range(rows)] for _ in range(cables)]


def build_store(root, images: int = 4, seed: int = 100) -> AnnotationStore:
    """Upload and completely mark a few boards, like an operator would."""
    rng = np.random.default_rng(seed)
    store = AnnotationStore(root)
    for i in range(images):
        img, boxes = render_marked_board(random_columns(rng), seed=seed + i, shift=tuple(rng.uniform(-20, 20, 2)))
        image_id, _ = store.add_image(img, name=f"board{i}.jpg")
        store.set_boxes(image_id, boxes)
    return store


@pytest.fixture(scope="session")
def marked_store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("annotations"))
