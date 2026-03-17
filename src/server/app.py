from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from src.server.routers.analyze import router as analyze_router
from src.server.routers.command import router as command_router
from src.server.routers.health import router as health_router
from src.server.state import AnalyzeState, AppState, RouteState


def create_app(upload_dir: str | Path = "data/interim/uploads") -> FastAPI:
    app = FastAPI(
        title="Vision AGV Integration Server",
        version="0.1.0",
        description="HTTP bridge between the ROS2 AGV client and the vision project.",
    )

    app.state.app_state = AppState(
        route_state=RouteState(),
        analyze_state=AnalyzeState(upload_dir=Path(upload_dir)),
    )

    app.include_router(health_router)
    app.include_router(command_router)
    app.include_router(analyze_router)
    return app


app = create_app()
