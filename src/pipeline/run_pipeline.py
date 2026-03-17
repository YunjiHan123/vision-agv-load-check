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
    """Run the end-to-end placeholder pipeline for one image or directory."""
    transformed = apply_perspective_transform(input_path)
    label_detections = detect_lot_labels(transformed, config_path=config_path)
    book_count = count_books_from_labels(label_detections)
    book_segments = segment_books_from_labels(transformed, label_detections)
    anomalies = detect_anomalies(book_segments, label_detections)
    lot_info = read_lot_information(transformed, label_detections)
    visualization = build_visualization_summary(book_count, anomalies, lot_info)

    return {
        "input_path": str(input_path),
        "config_path": str(config_path),
        "book_count": book_count,
        "labels": label_detections,
        "segments": book_segments,
        "anomalies": anomalies,
        "lot_info": lot_info,
        "visualization": visualization,
        "summary": (
            f"Pipeline placeholder completed for '{input_path}'. "
            f"Detected {book_count} label(s) and prepared downstream outputs."
        ),
    }
