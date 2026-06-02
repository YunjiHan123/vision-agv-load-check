from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.status import infer_status_from_path


def _extract_bbox_features(label_detections: list[dict[str, object]]) -> list[dict[str, float]]:
    features: list[dict[str, float]] = []
    for detection in label_detections:
        bbox = detection.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        features.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "cx": x1 + (width / 2.0),
                "cy": y1 + (height / 2.0),
                "bottom": y2,
                "aspect": width / height,
            }
        )
    return features


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _classify_status_from_bboxes(label_detections: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    features = _extract_bbox_features(label_detections)
    if not features:
        return "unknown", {}

    widths = [item["width"] for item in features]
    heights = [item["height"] for item in features]
    center_xs = [item["cx"] for item in features]
    bottoms = [item["bottom"] for item in features]
    center_ys = [item["cy"] for item in features]
    aspects = [item["aspect"] for item in features]

    mean_width = _mean(widths)
    mean_height = _mean(heights)
    mean_center_x = _mean(center_xs)
    x_offsets = [abs(value - mean_center_x) for value in center_xs]
    center_x_mean_abs_offset = _mean(x_offsets)
    center_x_span = max(center_xs) - min(center_xs) if len(center_xs) > 1 else 0.0
    center_y_span = max(center_ys) - min(center_ys) if len(center_ys) > 1 else 0.0
    bottom_range = max(bottoms) - min(bottoms) if len(bottoms) > 1 else 0.0
    wide_box_ratio = sum(1 for value in aspects if value > 1.1) / len(aspects)

    metrics = {
        "box_count": float(len(features)),
        "mean_width": mean_width,
        "mean_height": mean_height,
        "center_x_mean_abs_offset": center_x_mean_abs_offset,
        "center_x_span": center_x_span,
        "center_y_span": center_y_span,
        "bottom_range": bottom_range,
        "wide_box_ratio": wide_box_ratio,
    }

    horizontal_layout_ratio = center_x_span / max(center_y_span, 1.0)

    # Fallen scenes lose the vertical stack layout almost entirely.
    if (
        len(features) > 1
        and (
            horizontal_layout_ratio > 1.4
            or (center_x_span > mean_width * 1.35 and center_x_mean_abs_offset > mean_width * 0.34)
        )
    ):
        return "fallen", metrics

    # Misaligned scenes keep upright boxes but break the vertical center line.
    if (
        len(features) > 1
        and
        mean_width > 0
        and (
            center_x_mean_abs_offset > mean_width * 0.12
            or center_x_span > mean_width * 0.22
        )
    ):
        return "misaligned", metrics

    return "normal", metrics


def detect_anomalies(
    book_segments: list[dict[str, object]],
    label_detections: list[dict[str, object]],
    input_path: Path | None = None,
    perspective_result: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Classify stack status from YOLO bounding-box geometry, using filenames only as fallback."""
    predicted_status, metrics = _classify_status_from_bboxes(label_detections)
    if predicted_status == "unknown" and input_path is not None:
        predicted_status = infer_status_from_path(input_path)

    if predicted_status in {"normal", "unknown"}:
        return []

    perspective_applied = bool((perspective_result or {}).get("applied", False))
    return [
        {
            "status": predicted_status,
            "source": "bbox_geometry" if book_segments else "filename_fallback",
            "reference_view": "perspective_warp" if perspective_applied else "raw_only",
            "metrics": metrics,
        }
    ]


def classify_status_from_detections(label_detections: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    """Exported for tests and debugging."""
    return _classify_status_from_bboxes(label_detections)
