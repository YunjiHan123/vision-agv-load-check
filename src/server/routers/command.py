from __future__ import annotations

from fastapi import APIRouter, Request

from src.server.schemas import RouteResponse, RouteUpdateRequest, RouteUpdateResponse
from src.server.state import AppState

router = APIRouter(tags=["command"])


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


@router.get("/command", response_model=RouteResponse)
def get_command(request: Request) -> RouteResponse:
    state = get_app_state(request)
    return RouteResponse(route=state.route_state.route)


@router.post("/command", response_model=RouteUpdateResponse)
def update_command(payload: RouteUpdateRequest, request: Request) -> RouteUpdateResponse:
    state = get_app_state(request)
    snapshot = state.route_state.update(payload.route)
    return RouteUpdateResponse(**snapshot)
