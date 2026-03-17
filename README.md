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

- The current vision pipeline modules are still largely placeholders.
- The communication server is intentionally lightweight and keeps route state in memory for integration testing.
