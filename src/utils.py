"""Shared configuration, logging, and persistence utilities for PredictX."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration values used across the ML lifecycle."""

    random_seed: int = 42
    test_size: float = 0.2
    target_column: str = "Machine failure"
    categorical_columns: List[str] = field(default_factory=lambda: ["Type"])
    selected_features: List[str] = field(
        default_factory=lambda: [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    )
    raw_data_file: Path = DATA_RAW_DIR / "ai4i2020.csv"
    processed_data_file: Path = DATA_PROCESSED_DIR / "ai4i2020_processed.csv"
    data_quality_report_file: Path = DATA_PROCESSED_DIR / "data_quality_report.json"
    model_metrics_file: Path = MODELS_DIR / "model_metrics.json"
    best_model_file: Path = MODELS_DIR / "best_model.joblib"
    preprocessor_file: Path = MODELS_DIR / "preprocessor.joblib"
    predictions_schema_file: Path = MODELS_DIR / "prediction_schema.json"
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable configuration values."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def ensure_directories() -> None:
    """Create all project output directories used by pipelines."""
    for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Create a consistently formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    """Persist a dictionary as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning an empty dictionary if it is absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_artifact(obj: Any, path: Path) -> None:
    """Persist a Python artifact using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    """Load a persisted joblib artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
