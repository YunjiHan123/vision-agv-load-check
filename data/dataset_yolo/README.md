# YOLO Dataset Layout

Place labeled data here using YOLO detection format.

## Directory structure

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`

## Naming rule

Image and label filenames must share the same stem.

- `images/train/img_001.jpg`
- `labels/train/img_001.txt`

## Label format

Each line in a label file must follow:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All box coordinates must be normalized to `[0, 1]`.

## Current class map

- `0`: `mini_box`
