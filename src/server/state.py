from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RouteState:
    route: list[str] = field(default_factory=lambda: ["A", "D", "F"])
    updated_at: str = field(default_factory=utc_timestamp)

    def snapshot(self) -> dict[str, object]:
        return {
            "route": list(self.route),
            "updated_at": self.updated_at,
        }

    def update(self, route: list[str]) -> dict[str, object]:
        self.route = list(route)
        self.updated_at = utc_timestamp()
        return self.snapshot()


@dataclass
class AnalyzeState:
    upload_dir: Path
    last_upload_name: str = ""
    last_upload_bytes: int = 0
    last_upload_at: str = ""

    def snapshot(self) -> dict[str, object]:
        return {
            "last_upload_name": self.last_upload_name,
            "last_upload_bytes": self.last_upload_bytes,
            "last_upload_at": self.last_upload_at,
            "upload_dir": str(self.upload_dir),
        }

    def mark_upload(self, filename: str, size_bytes: int) -> dict[str, object]:
        self.last_upload_name = filename
        self.last_upload_bytes = size_bytes
        self.last_upload_at = utc_timestamp()
        return self.snapshot()


@dataclass
class AppState:
    route_state: RouteState
    analyze_state: AnalyzeState
