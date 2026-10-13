from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "test" / "model_blood_group_detection_resnet.h5").exists():
            return parent
        if (parent / "dataset" / "dataset_blood_group").exists():
            return parent
    return here.parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_find_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Goodbye to Blood Test"
    environment: str = "development"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str = "sqlite:///./backend/fingerabo.db"

    project_root: Path = Field(default_factory=_find_project_root)
    model_registry_dir: str = "models/registry"
    active_model_id: str = "resnet50_v1"
    resnet50_v1_path: str = "test/model_blood_group_detection_resnet.h5"
    image_size: int = 256
    max_upload_mb: int = 10
    store_prediction_images: bool = False
    temp_dir: str = "storage/tmp"

    class_labels: List[str] = Field(
        default_factory=lambda: ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def model_path(self) -> Path:
        return self.resolve(self.resnet50_v1_path)

    @property
    def registry_path(self) -> Path:
        return self.resolve(self.model_registry_dir)

    @property
    def temp_path(self) -> Path:
        return self.resolve(self.temp_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
