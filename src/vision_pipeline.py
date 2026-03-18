from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.run_pipeline import run_pipeline
from src.utils.status import normalize_status_label


def process_image(image_path: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Run the current placeholder pipeline and convert it to test CSV output."""
    image_path = Path(image_path)
    resolved_config = Path(config_path) if config_path is not None else Path("configs/pipeline.yaml")

    result = run_pipeline(input_path=image_path, config_path=resolved_config)

    anomalies = result.get("anomalies", [])
    lot_info = result.get("lot_info", [])
    pred_lots_list: list[str] = []

    if lot_info and isinstance(lot_info, list):
        for item in lot_info:
            if isinstance(item, dict) and item.get("lot"):
                pred_lots_list.append(str(item["lot"]))

    return {
        "pred_count": int(result.get("book_count", 0)),
        "pred_status": normalize_status_label(str(result.get("predicted_status", "normal" if not anomalies else "misaligned"))),
        "pred_lots": "|".join(pred_lots_list) if pred_lots_list else "UNKNOWN",
    }
