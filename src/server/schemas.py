from __future__ import annotations

from pydantic import BaseModel, Field


class RouteResponse(BaseModel):
    route: list[str] = Field(default_factory=list)


class RouteUpdateRequest(BaseModel):
    route: list[str] = Field(default_factory=list)


class RouteUpdateResponse(BaseModel):
    route: list[str] = Field(default_factory=list)
    updated_at: str


class HealthResponse(BaseModel):
    status: str


class AnalyzeResponse(BaseModel):
    filename: str
    size_bytes: int
    saved_path: str
    message: str
