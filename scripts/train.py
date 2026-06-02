from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_YAML = BASE_DIR / "data" / "dataset_yolo" / "dataset.yaml"
DEFAULT_PROJECT_DIR = BASE_DIR / "models" / "train_runs"
DEFAULT_OUTPUT_WEIGHTS = BASE_DIR / "models" / "mini_box_yolo.pt"
DEFAULT_PRETRAINED_WEIGHTS = "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for mini_box labels.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Path to dataset.yaml.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_PRETRAINED_WEIGHTS,
        help="Initial YOLO weights or model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device such as cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Directory where Ultralytics training runs are stored.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mini_box_detector",
        help="Run name inside the training project directory.",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=DEFAULT_OUTPUT_WEIGHTS,
        help="Final path where best.pt will be copied after training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    return parser.parse_args()


def count_images(image_dir: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def validate_dataset_layout(dataset_yaml: Path) -> None:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml}")

    dataset_root = dataset_yaml.parent
    expected_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]
    for directory in expected_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Required dataset directory is missing: {directory}")

    train_count = count_images(dataset_root / "images" / "train")
    val_count = count_images(dataset_root / "images" / "val")
    test_count = count_images(dataset_root / "images" / "test")
    if train_count == 0:
        raise ValueError("No training images found in images/train.")

    print("Dataset check")
    print(f"  root : {dataset_root}")
    print(f"  train: {train_count} image(s)")
    print(f"  val  : {val_count} image(s)")
    print(f"  test : {test_count} image(s)")


def resolve_best_weights(project_dir: Path, run_name: str) -> Path:
    best_weights = project_dir / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed but best.pt was not found: {best_weights}")
    return best_weights


def main() -> None:
    if YOLO is None:
        raise ModuleNotFoundError(
            "ultralytics is required for training. Install dependencies from requirements.txt first."
        ) from IMPORT_ERROR

    args = parse_args()
    data_path = args.data.resolve()
    project_dir = args.project.resolve()
    output_weights = args.output_weights.resolve()

    validate_dataset_layout(data_path)

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
    )

    best_weights = resolve_best_weights(project_dir, args.name)
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_weights)

    print("\nTraining finished")
    print(f"  run dir      : {project_dir / args.name}")
    print(f"  best weights : {best_weights}")
    print(f"  copied to    : {output_weights}")


if __name__ == "__main__":
    main()
