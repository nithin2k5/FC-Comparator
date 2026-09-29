"""Create part numbers from marked good boards."""

from __future__ import annotations

from .annotations import AnnotationStore
from .layout import master_from_boxes
from .models import PartNumber, Taxonomy


def part_from_marked_image(
    store: AnnotationStore, image_id: str, code: str, taxonomy: Taxonomy, description: str = "", cables: int | None = None
) -> tuple[PartNumber, list[str]]:
    """Build a part (pattern + layout) from a completely marked, known-good board image.

    Returns (part, notes). Raises ``LayoutError`` if the marking is not a complete,
    consistent good board, and ``ValueError`` for bad input.
    """
    code = code.strip()
    if not code:
        raise ValueError("Enter a part number")
    rec = store.get(image_id)
    if not rec.boxes:
        raise ValueError("This image has no marked clips yet")
    pattern, layout, notes = master_from_boxes(rec.boxes, (rec.width, rec.height), taxonomy, cables)
    part = PartNumber(code, pattern, description, cables=layout.cables, layout=layout, master_image=image_id)
    part.validate(taxonomy)
    return part, notes
