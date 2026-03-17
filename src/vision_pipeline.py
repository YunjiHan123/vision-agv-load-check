from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.run_pipeline import run_pipeline


def process_image(image_path: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Run the current placeholder pipeline and convert it to test CSV output."""
    image_path = Path(image_path)
    resolved_config = Path(config_path) if config_path is not None else Path("configs/pipeline.yaml")

    result = run_pipeline(input_path=image_path, config_path=resolved_config)

    anomalies = result.get("anomalies", [])
    lot_info = result.get("lot_info", [])
    pred_lot = "UNKNOWN"

    if lot_info and isinstance(lot_info, list):
        first_item = lot_info[0]
        if isinstance(first_item, dict):
            pred_lot = str(first_item.get("lot", "UNKNOWN"))

    return {
        "pred_count": int(result.get("book_count", 0)),
        "pred_status": "normal" if not anomalies else "abnormal",
        "pred_lot": pred_lot,
    }
