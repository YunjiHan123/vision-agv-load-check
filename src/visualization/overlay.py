from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _default_output_dir() -> Path:
    return Path("data/processed/overlays")


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "fallen":
        return (30, 30, 220)
    if status == "misaligned":
        return (0, 165, 255)
    return (0, 180, 0)


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top = max(y - height - 14, 0)
    cv2.rectangle(image, (x, top), (x + width + 12, y), color, thickness=-1)
    cv2.putText(image, text, (x + 6, y - 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _save_overlay_image(
    input_path: Path,
    raw_image: np.ndarray | None,
    detections: list[dict[str, object]],
    predicted_status: str,
    book_count: int,
) -> str | None:
    if raw_image is None:
        return None

    canvas = raw_image.copy()
    color = _status_color(predicted_status)

    for index, detection in enumerate(detections, start=1):
        bbox = detection.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=5)
        _draw_label(canvas, f"box {index}", (x1, max(y1, 40)), color)

    header_text = f"status={predicted_status} count={book_count}"
    cv2.rectangle(canvas, (20, 20), (640, 110), (25, 25, 25), thickness=-1)
    cv2.putText(canvas, header_text, (40, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_overlay.jpg"
    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def build_visualization_summary(
    book_count: int,
    anomalies: list[dict[str, object]],
    lot_info: list[dict[str, object]],
    perspective_result: dict[str, object] | None = None,
    input_path: Path | None = None,
    raw_image: np.ndarray | None = None,
    detections: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Build visualization metadata and save an overlay image when possible."""
    predicted_status = "normal" if not anomalies else str(anomalies[0].get("status", "misaligned"))
    overlay_path = None
    if input_path is not None:
        overlay_path = _save_overlay_image(
            input_path=input_path,
            raw_image=raw_image,
            detections=detections or [],
            predicted_status=predicted_status,
            book_count=book_count,
        )

    return {
        "book_count": book_count,
        "anomaly_count": len(anomalies),
        "lot_records": len(lot_info),
        "predicted_status": predicted_status,
        "perspective_applied": bool((perspective_result or {}).get("applied", False)),
        "perspective_status": (perspective_result or {}).get("status", "not_run"),
        "overlay_path": overlay_path,
    }
