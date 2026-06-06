"""Unit tests for PredictX prediction pipeline."""
from __future__ import annotations

from pathlib import Path

from src.data_preprocessing import create_sample_ai4i_dataset, preprocess_data
from src.predict import MachineInput, predict_failure, risk_category
from src.train import train_models
from src.utils import ProjectConfig


def test_risk_category_thresholds() -> None:
    assert risk_category(0.1) == "Low"
    assert risk_category(0.25) == "Medium"
    assert risk_category(0.45) == "High"
    assert risk_category(0.75) == "Critical"


def test_predict_failure_returns_operational_decision(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    schema_path = tmp_path / "schema.json"
    create_sample_ai4i_dataset(n_rows=260).to_csv(raw_path, index=False)
    config = ProjectConfig(
        raw_data_file=raw_path,
        processed_data_file=processed_path,
        best_model_file=model_path,
        model_metrics_file=metrics_path,
        preprocessor_file=preprocessor_path,
        predictions_schema_file=schema_path,
    )
    processed = preprocess_data(input_path=raw_path, output_path=processed_path, config=config)
    train_models(processed, config)
    result = predict_failure(MachineInput("L", 300.0, 310.0, 1500, 40.0, 100), config=config)
    assert {"failure_prediction", "failure_probability", "risk_category", "recommended_action"}.issubset(result)
    assert 0.0 <= result["failure_probability"] <= 1.0
