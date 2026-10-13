from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(512), nullable=False)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=256)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CANDIDATE")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    split_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    image_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    predicted_label: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_id: Mapped[str] = mapped_column(String(64), ForeignKey("model_versions.id"), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped[ModelVersion] = relationship(back_populates="predictions")
    probabilities: Mapped[list["PredictionProbability"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class PredictionProbability(Base):
    __tablename__ = "prediction_probabilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), ForeignKey("predictions.id"), nullable=False)
    label: Mapped[str] = mapped_column(String(8), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    prediction: Mapped[Prediction] = relationship(back_populates="probabilities")
