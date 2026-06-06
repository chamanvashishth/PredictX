"""Unit tests for PredictX preprocessing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_preprocessing import clean_data, create_sample_ai4i_dataset, encode_features, preprocess_data
from src.utils import ProjectConfig


def test_clean_data_removes_duplicates_and_missing_values() -> None:
    df = create_sample_ai4i_dataset(n_rows=25)
    df.loc[0, "Torque [Nm]"] = None
    dirty = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    clean = clean_data(dirty)
    assert clean.duplicated().sum() == 0
    assert clean.isna().sum().sum() == 0


def test_encode_features_converts_type_to_numeric() -> None:
    df = create_sample_ai4i_dataset(n_rows=50)
    encoded, encoders = encode_features(clean_data(df))
    assert "Type" in encoders
    assert pd.api.types.is_integer_dtype(encoded["Type"])


def test_preprocess_data_exports_processed_dataset(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    create_sample_ai4i_dataset(n_rows=60).to_csv(raw_path, index=False)
    config = ProjectConfig(raw_data_file=raw_path, processed_data_file=output_path)
    processed = preprocess_data(input_path=raw_path, output_path=output_path, config=config)
    assert output_path.exists()
    assert set(config.selected_features + [config.target_column]).issubset(processed.columns)
