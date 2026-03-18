from __future__ import annotations

from pathlib import Path

from src.utils.status import infer_status_from_path


def detect_anomalies(
    book_segments: list[dict[str, object]],
    label_detections: list[dict[str, object]],
    input_path: Path | None = None,
    perspective_result: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Placeholder anomaly rules using raw image status, with perspective as auxiliary context."""
    predicted_status = "unknown"
    if label_detections:
        predicted_status = str(label_detections[0].get("scene_status_hint", "unknown"))
    if predicted_status == "unknown" and input_path is not None:
        predicted_status = infer_status_from_path(input_path)
    if predicted_status in {"normal", "unknown"}:
        return []

    perspective_applied = bool((perspective_result or {}).get("applied", False))
    return [
        {
            "status": predicted_status,
            "source": "raw_image_primary" if book_segments else "filename_fallback",
            "reference_view": "perspective_warp" if perspective_applied else "raw_only",
        }
    ]
