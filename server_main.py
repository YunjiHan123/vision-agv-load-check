from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("VISION_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("VISION_SERVER_PORT", "8000"))
    uvicorn.run("src.server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
