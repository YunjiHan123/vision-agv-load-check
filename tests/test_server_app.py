from __future__ import annotations

from fastapi.testclient import TestClient

from src.server.app import create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app(upload_dir="data/interim/test_uploads"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_command_endpoints_round_trip_route() -> None:
    client = TestClient(create_app(upload_dir="data/interim/test_uploads"))

    initial_response = client.get("/command")
    update_response = client.post("/command", json={"route": ["B", "C", "E"]})
    refreshed_response = client.get("/command")

    assert initial_response.status_code == 200
    assert initial_response.json()["route"] == ["A", "D", "F"]
    assert update_response.status_code == 200
    assert update_response.json()["route"] == ["B", "C", "E"]
    assert refreshed_response.status_code == 200
    assert refreshed_response.json() == {"route": ["B", "C", "E"]}


def test_analyze_endpoint_accepts_file_upload(tmp_path) -> None:
    client = TestClient(create_app(upload_dir=tmp_path))

    response = client.post(
        "/analyze",
        files={"file": ("frame.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "frame.jpg"
    assert body["size_bytes"] == len(b"fake-image-bytes")
    assert (tmp_path / "frame.jpg").exists()
