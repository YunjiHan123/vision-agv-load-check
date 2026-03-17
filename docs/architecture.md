# Architecture

## Purpose

This repository combines two related but separate concerns:

1. Vision pipeline and evaluation code
2. HTTP server for ROS2 AGV integration

Keeping them in one repository is acceptable as long as runtime boundaries stay clear.

## Runtime Boundaries

### Pipeline / Test Program

- entrypoint: `main.py`
- main responsibility: run the current pipeline skeleton for local vision experiments
- related modules: `src/pipeline`, `src/detection`, `src/ocr`, `src/evaluate.py`, `src/run_test.py`

### Integration Server

- entrypoint: `server_main.py`
- app factory: `src/server/app.py`
- main responsibility: expose HTTP endpoints used by the ROS2 AGV client
- related modules: `src/server/routers`, `src/server/state.py`, `src/server/schemas.py`

## HTTP Contract

### AGV -> Vision Server

- `POST /analyze`
- request: `multipart/form-data`
- file field: `file`
- current behavior: save uploaded file and return metadata

### Vision Server -> AGV Polling Contract

- `GET /command`
- response example:

```json
{
  "route": ["A", "D", "F"]
}
```

### Test / Dashboard Route Update

- `POST /command`
- request example:

```json
{
  "route": ["B", "C", "E"]
}
```

### Health Check

- `GET /health`

## State Management

- route state is kept in memory
- uploaded images are stored under `data/interim/uploads`
- this is enough for communication validation with the ROS team
- persistent storage or vision inference hooks can be added later without changing the external API
