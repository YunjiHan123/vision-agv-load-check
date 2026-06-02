from __future__ import annotations

from pathlib import Path

from src.pipeline.run_pipeline import run_pipeline


def test_trained_detector_runs_on_sample_images() -> None:
    sample_images = ["img_001.jpg", "img_004.jpg", "img_006.jpg"]

    for image_name in sample_images:
        result = run_pipeline(
            input_path=Path("data/test_images") / image_name,
            config_path=Path("configs/pipeline.yaml"),
        )

        assert result["book_count"] >= 1
        assert result["labels"]
        assert result["labels"][0]["detector_source"] in {"ultralytics", "heuristic_fallback"}
