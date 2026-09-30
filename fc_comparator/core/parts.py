"""Create a part number (master pattern + layout) from the clips on a known-good board."""

from __future__ import annotations

from .layout import master_from_boxes
from .models import Box, PartNumber, Taxonomy


def part_from_boxes(
    code: str,
    boxes: list[Box],
    image_size: tuple[int, int],
    taxonomy: Taxonomy,
    description: str = "",
    master_image: str = "",
) -> PartNumber:
    """``boxes`` are the clips of a good board - detected or marked by hand.

    Raises ``LayoutError`` if they do not form a complete cables x rows board and
    ``ValueError`` for bad input.
    """
    code = code.strip()
    if not code:
        raise ValueError("Enter a part number")
    if not boxes:
        raise ValueError("No clips on this board")
    pattern, layout = master_from_boxes(boxes, image_size)
    part = PartNumber(code, pattern, description.strip(), layout=layout, master_image=master_image)
    part.validate(taxonomy)
    return part
