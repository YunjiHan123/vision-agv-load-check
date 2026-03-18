from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
import yaml

from src.preprocess.perspective import apply_perspective_transform


def _write_test_image(image_path: Path) -> None:
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    source_points = np.array([[50, 40], [170, 60], [160, 180], [60, 170]], dtype=np.int32)
    cv2.fillConvexPoly(image, source_points, color=(0, 255, 0))
    cv2.imwrite(str(image_path), image)


def _make_test_dir() -> Path:
    test_dir = Path("data/interim/test_perspective") / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    return test_dir


def test_perspective_transform_applies_sidecar_points() -> None:
    test_dir = _make_test_dir()
    image_path = test_dir / "sample.png"
    output_dir = test_dir / "processed"
    _write_test_image(image_path)

    try:
        config_path = test_dir / "sample.perspective.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "source_points": [[50, 40], [170, 60], [160, 180], [60, 170]],
                    "output_size": [120, 140],
                }
            ),
            encoding="utf-8",
        )

        result = apply_perspective_transform(image_path, output_dir=output_dir)

        assert result["status"] == "applied"
        assert result["applied"] is True
        assert result["warped_shape"][:2] == [140, 120]
        assert Path(result["warped_path"]).exists()
        assert Path(result["visualization_path"]).exists()

        warped_image = result["image"]
        center_patch = warped_image[50:90, 40:80]
        assert float(center_patch[..., 1].mean()) > 200.0
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_perspective_transform_falls_back_without_points() -> None:
    test_dir = _make_test_dir()
    image_path = test_dir / "sample.png"
    output_dir = test_dir / "processed"
    _write_test_image(image_path)

    try:
        result = apply_perspective_transform(image_path, output_dir=output_dir)

        assert result["status"] == "points_not_defined"
        assert result["applied"] is False
        assert result["warped_path"] is None
        assert result["visualization_path"] is None
        assert result["image"].shape == (220, 220, 3)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
