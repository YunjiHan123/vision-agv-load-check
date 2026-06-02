# Vision AGV Load Check

This repository now serves two purposes:

- vision pipeline and evaluation/test tooling
- FastAPI communication server for ROS2 AGV integration

## Runtime Split

### 1. Pipeline / Test Program

Use the existing CLI entrypoint to run the current vision pipeline skeleton:

```bash
python main.py --input data/samples
```

This path is for local pipeline experiments, mock-based testing, and evaluation scripts.

#### OCR Environment

The OCR module supports two runtime paths:

- default `.venv`: `PaddleOCR -> EasyOCR fallback`
- strict `.venv-paddle`: `PaddleOCR only`

Run tests and end-to-end evaluation in the default environment:

```bash
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe src\run_test.py --mode real
.venv\Scripts\python.exe src\evaluate.py
```

#### PaddleOCR-Only Setup

For a strict PaddleOCR-only run on this Windows machine, use Python `3.12` with `numpy 1.26.4`:

```bash
uv venv --python 3.12 .venv-paddle
uv pip install --python .venv-paddle\Scripts\python.exe -r requirements-paddle.txt
```

Use an ASCII-only Paddle cache path because Paddle inference can fail on non-ASCII model paths:

```powershell
$env:HOME='C:\Users\han\.book_lot_vision_cache\home'
$env:USERPROFILE=$env:HOME
$env:XDG_CACHE_HOME='C:\Users\han\.book_lot_vision_cache\home\.cache'
$env:PADDLE_HOME='C:\Users\han\.book_lot_vision_cache\home\.cache\paddle'
$env:PADDLE_PDX_CACHE_HOME='C:\Users\han\.book_lot_vision_cache\paddlex'
.venv-paddle\Scripts\python.exe -m pytest -q
```

The Paddle-only config also enables OCR-specific crop tuning:

- expanded crop padding
- deskew-based rotation
- small-angle rotation search
- valid lot pool snapping

To force PaddleOCR without EasyOCR fallback, use:

```bash
.venv-paddle\Scripts\python.exe src\run_test.py --mode real --config configs/ocr_paddle_only.yaml
.venv-paddle\Scripts\python.exe src\evaluate.py
```

### 2. ROS Integration Server

Use the server entrypoint to run the HTTP server expected by the ROS2 AGV node:

```bash
python server_main.py
```

Default server address:

- host: `0.0.0.0`
- port: `8000`

Environment variables:

- `VISION_SERVER_HOST`
- `VISION_SERVER_PORT`

## Supported Endpoints

- `GET /health`
- `GET /command`
- `POST /command`
- `POST /analyze`

`POST /command` is the test/dashboard route update endpoint. `GET /command` is the endpoint polled by the AGV.

## Project Layout

```text
vision-agv-load-check/
  configs/
  data/
  docs/
  src/
    pipeline/
    server/
  tests/
  main.py
  server_main.py
```

## Notes

- `src/ocr/lot_reader.py` now expands YOLO mini-box crops before OCR and tests multiple rotated variants before choosing the best lot code.
- `configs/ocr.yaml` and `configs/ocr_paddle_only.yaml` define OCR padding, deskew, rotation search, and valid lot pool behavior.
- The communication server is intentionally lightweight and keeps route state in memory for integration testing.
