from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from src.server.schemas import AnalyzeResponse
from src.server.state import AppState

router = APIRouter(tags=["analyze"])


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def build_upload_path(upload_dir: Path, original_name: str) -> Path:
    safe_name = Path(original_name or "upload.bin").name
    target = upload_dir / safe_name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    index = 1
    while True:
        candidate = upload_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
) -> AnalyzeResponse:
    state = get_app_state(request)
    upload_dir = state.analyze_state.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    target_path = build_upload_path(upload_dir, file.filename or "upload.bin")
    target_path.write_bytes(content)
    size_bytes = len(content)
    state.analyze_state.mark_upload(filename=target_path.name, size_bytes=size_bytes)

    return AnalyzeResponse(
        filename=target_path.name,
        size_bytes=size_bytes,
        saved_path=str(target_path),
        message="Image received successfully.",
    )
