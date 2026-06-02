from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision_pipeline import process_image
from src.utils.status import normalize_status_label
from src.utils.lot_codes import LOT_POOL
from src.utils.project_paths import GROUND_TRUTH_CSV, PIPELINE_CONFIG_PATH, RESULT_CSV, TEST_IMAGES_DIR


IMAGE_DIR = TEST_IMAGES_DIR
GT_CSV = GROUND_TRUTH_CSV
CONFIG_PATH = PIPELINE_CONFIG_PATH
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results.csv from images or mock data.")
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="auto: use real images if present, otherwise use mock predictions.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Configuration file used for real pipeline inference.",
    )
    return parser.parse_args()


def iter_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []

    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_ground_truth_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_ground_truth_map(csv_path: Path) -> dict[str, dict[str, str]]:
    return {row["image_name"]: row for row in load_ground_truth_rows(csv_path)}


def split_lots(lot_text: str) -> list[str]:
    return [token.strip() for token in lot_text.split("|") if token.strip()]


def join_lots(lots: list[str]) -> str:
    return "|".join(lots)


def rotate_lot(lot_code: str) -> str:
    if lot_code not in LOT_POOL:
        return LOT_POOL[0]
    return LOT_POOL[(LOT_POOL.index(lot_code) + 1) % len(LOT_POOL)]


def build_mock_prediction(image_name: str) -> dict[str, str | int]:
    """Return deterministic mock outputs with a mix of exact and imperfect cases."""
    gt_row = load_ground_truth_map(GT_CSV).get(image_name)
    if gt_row is None:
        return {"pred_count": 0, "pred_status": "normal", "pred_lots": "UNKNOWN"}

    image_number = int(Path(image_name).stem.split("_")[-1])
    pred_count = int(gt_row["true_count"])
    pred_status = normalize_status_label(gt_row["true_status"])
    pred_lots_list = split_lots(gt_row["true_lots"])
    scenario = image_number % 10

    if scenario == 2:
        pred_count = max(0, pred_count - 1)
    elif scenario == 3:
        if pred_lots_list:
            pred_lots_list[-1] = rotate_lot(pred_lots_list[-1])
    elif scenario == 4:
        pred_status = "fallen" if pred_status == "normal" else "normal"
    elif scenario == 6:
        pred_count = min(5, pred_count + 1)
    elif scenario == 8:
        if len(pred_lots_list) > 1:
            pred_lots_list = pred_lots_list[:-1]
    elif scenario == 9:
        pred_status = "normal"

    return {
        "pred_count": pred_count,
        "pred_status": normalize_status_label(pred_status),
        "pred_lots": join_lots(pred_lots_list) if pred_lots_list else "UNKNOWN",
    }


def generate_mock_results() -> int:
    gt_rows = load_ground_truth_rows(GT_CSV)

    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_count", "pred_status", "pred_lots"])

        for row in gt_rows:
            prediction = build_mock_prediction(row["image_name"])
            writer.writerow(
                [
                    row["image_name"],
                    prediction["pred_count"],
                    prediction["pred_status"],
                    prediction["pred_lots"],
                ]
            )
            print(
                f"[MOCK] {row['image_name']} -> "
                f"count={prediction['pred_count']}, "
                f"status={prediction['pred_status']}, "
                f"lots={prediction['pred_lots']}"
            )

    return len(gt_rows)


def generate_real_results(image_files: list[Path], config_path: Path = CONFIG_PATH) -> int:
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_count", "pred_status", "pred_lots"])

        for image_path in image_files:
            result = process_image(image_path=image_path, config_path=config_path)
            writer.writerow(
                [
                    image_path.name,
                    result["pred_count"],
                    result["pred_status"],
                    result["pred_lots"],
                ]
            )
            print(
                f"[DONE] {image_path.name} -> "
                f"count={result['pred_count']}, "
                f"status={result['pred_status']}, "
                f"lots={result['pred_lots']}"
            )

    return len(image_files)


def main() -> None:
    args = parse_args()
    image_files = iter_image_files(IMAGE_DIR)
    should_use_mock = args.mode == "mock" or (args.mode == "auto" and not image_files)

    if args.mode == "real" and not image_files:
        raise FileNotFoundError(f"No test images found in: {IMAGE_DIR}")

    if should_use_mock:
        processed_count = generate_mock_results()
        print("mode: mock")
    else:
        processed_count = generate_real_results(image_files, config_path=args.config)
        print("mode: real")

    print(f"results.csv generated: {RESULT_CSV}")
    print(f"processed items: {processed_count}")


if __name__ == "__main__":
    main()
