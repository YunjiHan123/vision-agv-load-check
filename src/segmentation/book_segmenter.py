from __future__ import annotations


def segment_books_from_labels(
    image: dict[str, object],
    label_detections: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build lightweight per-box segments from detector outputs."""
    _ = image
    segments: list[dict[str, object]] = []
    for index, detection in enumerate(label_detections):
        segments.append(
            {
                "segment_id": index,
                "bbox": detection.get("bbox", []),
                "status_hint": detection.get("scene_status_hint", "normal"),
                "source": detection.get("detector_source", "unknown"),
            }
        )
    return segments
