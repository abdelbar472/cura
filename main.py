"""Compatibility entry point so ``uvicorn main:app`` keeps working."""

from __future__ import annotations

import uvicorn

from app.config import get_settings
from app.main import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.environment == "dev",
    )
