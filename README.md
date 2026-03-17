# Book Lot Vision

Python-based machine vision project skeleton for:

- inventory quantity estimation via lot label counting
- book state anomaly detection via label-guided segmentation
- lot information extraction with OCR

## Pipeline

1. Perspective transform
2. YOLO `lot_label` detection
3. Label count to book count
4. Label-position-based book segmentation
5. Rule-based anomaly detection
6. OCR-based lot parsing

## Project Layout

```text
book_lot_vision/
  configs/
  data/
  docs/
  models/
  notebooks/
  scripts/
  src/
  tests/
  main.py
```

## Quick Start

```bash
python main.py --input data/samples
```

The current version provides project structure and executable placeholders only.
