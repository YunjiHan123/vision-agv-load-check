from __future__ import annotations

from pathlib import Path
from typing import Any

from src.anomaly.rules import detect_anomalies
from src.counting.label_counter import count_books_from_labels
from src.detection.yolo_detector import detect_lot_labels
from src.ocr.lot_reader import read_lot_information
from src.preprocess.perspective import apply_perspective_transform
from src.segmentation.book_segmenter import segment_books_from_labels
from src.visualization.overlay import build_visualization_summary


def run_pipeline(input_path: Path, config_path: Path) -> dict[str, Any]:
    """Run the placeholder pipeline with raw-image primary analysis and perspective as auxiliary context."""
    perspective_result = apply_perspective_transform(input_path)
    analysis_views = {
        "raw_input_path": str(input_path),
        "raw_image": perspective_result.get("original_image"),
        "reference_image": perspective_result.get("image"),
        "perspective": perspective_result,
    }

    label_detections = detect_lot_labels(analysis_views, config_path=config_path)
    book_count = count_books_from_labels(label_detections)
    book_segments = segment_books_from_labels(analysis_views, label_detections)
    anomalies = detect_anomalies(
        book_segments,
        label_detections,
        input_path=input_path,
        perspective_result=perspective_result,
    )
    lot_info = read_lot_information(analysis_views, label_detections)
    visualization = build_visualization_summary(
        book_count,
        anomalies,
        lot_info,
        perspective_result=perspective_result,
        input_path=input_path,
        raw_image=perspective_result.get("original_image"),
        detections=label_detections,
    )
    predicted_status = "normal" if not anomalies else str(anomalies[0].get("status", "misaligned"))

    return {
        "input_path": str(input_path),
        "config_path": str(config_path),
        "analysis_views": analysis_views,
        "book_count": book_count,
        "labels": label_detections,
        "segments": book_segments,
        "anomalies": anomalies,
        "predicted_status": predicted_status,
        "lot_info": lot_info,
        "visualization": visualization,
        "summary": (
            f"Pipeline placeholder completed for '{input_path}'. "
            f"Raw image is the primary view, perspective warp is auxiliary "
            f"({perspective_result.get('status')}); detected {book_count} label(s)."
        ),
    }
