from __future__ import annotations

from pathlib import Path

from src.pipeline.run_pipeline import run_pipeline


def test_pipeline_placeholder_runs() -> None:
    result = run_pipeline(
        input_path=Path("data/samples"),
        config_path=Path("configs/pipeline.yaml"),
    )

    assert result["book_count"] == 0
    assert result["anomalies"] == []
    assert result["lot_info"] == []
    assert "summary" in result


def test_pipeline_normalizes_shifted_and_tilted_to_misaligned() -> None:
    result = run_pipeline(
        input_path=Path("data/test_images/shifted_001.jpg"),
        config_path=Path("configs/pipeline.yaml"),
    )

    assert result["predicted_status"] == "misaligned"
    assert result["anomalies"][0]["status"] == "misaligned"
