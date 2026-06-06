"""Unit tests for PredictX training pipeline."""
from __future__ import annotations

from pathlib import Path

from src.data_preprocessing import create_sample_ai4i_dataset, preprocess_data
from src.train import train_models
from src.utils import ProjectConfig


def test_train_models_returns_metrics_and_best_model(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    schema_path = tmp_path / "schema.json"
    create_sample_ai4i_dataset(n_rows=240).to_csv(raw_path, index=False)
    config = ProjectConfig(
        raw_data_file=raw_path,
        processed_data_file=processed_path,
        best_model_file=model_path,
        model_metrics_file=metrics_path,
        preprocessor_file=preprocessor_path,
        predictions_schema_file=schema_path,
    )
    processed = preprocess_data(input_path=raw_path, output_path=processed_path, config=config)
    models, metrics, best = train_models(processed, config)
    assert best in models
    assert model_path.exists()
    assert all("f1" in model_metrics for model_metrics in metrics.values())
