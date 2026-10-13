from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    request_id: str
    prediction: str = Field(description="Model prediction label (not a confirmed blood group)")
    confidence: float
    probabilities: Dict[str, float]
    top3: List[ProbabilityItem]
    model_name: str
    model_version: str
    model_id: str
    preprocessing_version: str
    image_hash: str
    timestamp: datetime
    disclaimer: str


class ModelInfo(BaseModel):
    id: str
    model_name: str
    version: str
    architecture: str
    status: str
    is_active: bool
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    split_type: Optional[str] = None
    image_size: int
    notes: Optional[str] = None


class DashboardSummary(BaseModel):
    app_name: str
    total_dataset_images: int
    n_classes: int
    deployed_model_name: Optional[str]
    deployed_model_version: Optional[str]
    deployed_model_accuracy: Optional[float]
    deployed_model_macro_f1: Optional[float]
    total_predictions: int
    total_experiments: int
    leakage_status: str
    class_distribution: Dict[str, int]
    disclaimer: str
