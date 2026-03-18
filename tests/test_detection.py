from __future__ import annotations

from pathlib import Path

from src.pipeline.run_pipeline import run_pipeline


def test_test_images_produce_expected_placeholder_statuses() -> None:
    expected = {
        "fallen_001.jpg": ("fallen", 4),
        "normal_001.jpg": ("normal", 4),
        "shifted_001.jpg": ("misaligned", 4),
        "tilted_001.jpg": ("misaligned", 4),
    }

    for image_name, (status, count) in expected.items():
        result = run_pipeline(
            input_path=Path("data/test_images") / image_name,
            config_path=Path("configs/pipeline.yaml"),
        )

        assert result["predicted_status"] == status
        assert result["book_count"] == count
        assert result["labels"]
