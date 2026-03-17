from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.vision_pipeline import process_image


IMAGE_DIR = BASE_DIR / "data" / "test_images"
GT_CSV = BASE_DIR / "data" / "ground_truth.csv"
RESULT_CSV = BASE_DIR / "data" / "results.csv"
CONFIG_PATH = BASE_DIR / "configs" / "pipeline.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results.csv from images or mock data.")
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="auto: use real images if present, otherwise use mock predictions.",
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


def build_mock_prediction(image_name: str) -> dict[str, str | int]:
    """Return fixed mock outputs that intentionally mix TP/FP/FN/TN cases."""
    mock_predictions: dict[str, dict[str, str | int]] = {
        "img_001.jpg": {"pred_count": 2, "pred_status": "normal", "pred_lot": "A10001"},
        "img_002.jpg": {"pred_count": 2, "pred_status": "normal", "pred_lot": "A10002"},
        "img_003.jpg": {"pred_count": 4, "pred_status": "shifted", "pred_lot": "A10033"},
        "img_004.jpg": {"pred_count": 5, "pred_status": "fallen", "pred_lot": "A10004"},
        "img_005.jpg": {"pred_count": 3, "pred_status": "tilted", "pred_lot": "A10005"},
        "img_006.jpg": {"pred_count": 5, "pred_status": "shifted", "pred_lot": "A10006"},
        "img_007.jpg": {"pred_count": 3, "pred_status": "normal", "pred_lot": "A10007"},
        "img_008.jpg": {"pred_count": 4, "pred_status": "shifted", "pred_lot": "A10888"},
        "img_009.jpg": {"pred_count": 1, "pred_status": "normal", "pred_lot": "A10009"},
        "img_010.jpg": {"pred_count": 5, "pred_status": "fallen", "pred_lot": "A10010"},
    }
    default_prediction = {"pred_count": 0, "pred_status": "normal", "pred_lot": "UNKNOWN"}
    return mock_predictions.get(image_name, default_prediction)


def generate_mock_results() -> int:
    gt_rows = load_ground_truth_rows(GT_CSV)

    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_count", "pred_status", "pred_lot"])

        for row in gt_rows:
            prediction = build_mock_prediction(row["image_name"])
            writer.writerow(
                [
                    row["image_name"],
                    prediction["pred_count"],
                    prediction["pred_status"],
                    prediction["pred_lot"],
                ]
            )
            print(
                f"[MOCK] {row['image_name']} -> "
                f"count={prediction['pred_count']}, "
                f"status={prediction['pred_status']}, "
                f"lot={prediction['pred_lot']}"
            )

    return len(gt_rows)


def generate_real_results(image_files: list[Path]) -> int:
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_count", "pred_status", "pred_lot"])

        for image_path in image_files:
            result = process_image(image_path=image_path, config_path=CONFIG_PATH)
            writer.writerow(
                [
                    image_path.name,
                    result["pred_count"],
                    result["pred_status"],
                    result["pred_lot"],
                ]
            )
            print(
                f"[DONE] {image_path.name} -> "
                f"count={result['pred_count']}, "
                f"status={result['pred_status']}, "
                f"lot={result['pred_lot']}"
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
        processed_count = generate_real_results(image_files)
        print("mode: real")

    print(f"results.csv generated: {RESULT_CSV}")
    print(f"processed items: {processed_count}")


if __name__ == "__main__":
    main()
