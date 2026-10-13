from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.services.registry import DISCLAIMER, ensure_resnet50_v1_registered


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.temp_path.mkdir(parents=True, exist_ok=True)
    settings.registry_path.mkdir(parents=True, exist_ok=True)
    Path(settings.project_root / "storage").mkdir(parents=True, exist_ok=True)
    init_db()
    db = SessionLocal()
    try:
        ensure_resnet50_v1_registered(db, settings)
    finally:
        db.close()
    # Eager-load model so first prediction is faster.
    from app.ml.resnet50_service import get_active_model_service

    get_active_model_service()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description=(
        "Experimental fingerprint → ABO/Rh research API. "
        + DISCLAIMER
    ),
    version="0.1.0-phase1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "phase": 1,
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }
