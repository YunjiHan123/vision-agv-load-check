from __future__ import annotations


def count_books_from_labels(label_detections: list[dict[str, object]]) -> int:
    """Use label count as book count."""
    return len(label_detections)
