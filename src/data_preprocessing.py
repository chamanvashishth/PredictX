"""Data ingestion, validation, cleaning, and feature preparation for AI4I 2020."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import ProjectConfig, ensure_directories, get_logger, save_artifact, save_json

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()

REQUIRED_COLUMNS = CONFIG.selected_features + [CONFIG.target_column]


def create_sample_ai4i_dataset(n_rows: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a statistically realistic AI4I-like fallback dataset for offline use."""
    rng = np.random.default_rng(random_seed)
    product_type = rng.choice(["L", "M", "H"], size=n_rows, p=[0.6, 0.3, 0.1])
    air_temp = rng.normal(300, 2.0, n_rows).round(2)
    process_temp = (air_temp + rng.normal(10, 1.0, n_rows)).round(2)
    speed = rng.normal(1538, 175, n_rows).clip(1150, 2900).round(0)
    torque = rng.normal(40, 10, n_rows).clip(3, 80).round(2)
    tool_wear = rng.integers(0, 254, n_rows)
    thermal_stress = np.maximum(process_temp - air_temp - 10, 0)
    mechanical_load = (torque / 55) + (tool_wear / 240) + np.maximum(1450 - speed, 0) / 600
    logits = -5.2 + 0.8 * thermal_stress + 1.6 * mechanical_load + 0.015 * (tool_wear - 120)
    probability = 1 / (1 + np.exp(-logits))
    failures = rng.binomial(1, np.clip(probability, 0.01, 0.45))
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n_rows + 1),
            "Product ID": [f"P-{i:05d}" for i in range(1, n_rows + 1)],
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": speed.astype(int),
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Machine failure": failures,
        }
    )


def fetch_dataset(config: ProjectConfig = CONFIG) -> Path:
    """Download AI4I 2020 data when possible, falling back to a deterministic local dataset."""
    ensure_directories()
    if config.raw_data_file.exists():
        return config.raw_data_file
    try:
        LOGGER.info("Downloading AI4I dataset from %s", config.dataset_url)
        df = pd.read_csv(config.dataset_url)
    except Exception as exc:  # network outages should not block local reproducibility
        LOGGER.warning("Dataset download failed (%s). Creating deterministic offline fallback.", exc)
        df = create_sample_ai4i_dataset(random_seed=config.random_seed)
    df.to_csv(config.raw_data_file, index=False)
    return config.raw_data_file


def load_raw_data(path: Path | None = None, config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Load raw AI4I data from disk, downloading it if missing."""
    data_path = path or fetch_dataset(config)
    LOGGER.info("Loading raw data from %s", data_path)
    return pd.read_csv(data_path)


def validate_features(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate that all required modeling fields are present and numeric fields are sane."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    numeric_columns = [column for column in required_columns if column not in {"Type", CONFIG.target_column}]
    invalid_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if invalid_numeric:
        raise TypeError(f"Numeric columns have invalid dtypes: {invalid_numeric}")
    if not set(df[CONFIG.target_column].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column must contain binary labels 0/1.")


def build_data_quality_report(df: pd.DataFrame) -> Dict[str, object]:
    """Create a compact data quality profile for governance and debugging."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            str(key): int(value) for key, value in df[CONFIG.target_column].value_counts().sort_index().items()
        },
        "numeric_summary": df.select_dtypes(include="number").describe().round(3).to_dict(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and impute missing values with robust defaults."""
    validate_features(df)
    clean = df.copy().drop_duplicates()
    for column in CONFIG.selected_features:
        if clean[column].isna().any():
            if pd.api.types.is_numeric_dtype(clean[column]):
                clean[column] = clean[column].fillna(clean[column].median())
            else:
                clean[column] = clean[column].fillna(clean[column].mode().iloc[0])
    clean[CONFIG.target_column] = clean[CONFIG.target_column].astype(int)
    return clean


def encode_features(df: pd.DataFrame, config: ProjectConfig = CONFIG) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode configured categorical features and persist encoder metadata."""
    encoded = df[config.selected_features + [config.target_column]].copy()
    encoders: Dict[str, LabelEncoder] = {}
    for column in config.categorical_columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder
    return encoded, encoders


def preprocess_data(
    input_path: Path | None = None,
    output_path: Path | None = None,
    config: ProjectConfig = CONFIG,
) -> pd.DataFrame:
    """Run the complete preprocessing workflow and export the processed dataset."""
    ensure_directories()
    raw_df = load_raw_data(input_path, config)
    validate_features(raw_df)
    quality_report = build_data_quality_report(raw_df)
    save_json(quality_report, config.data_quality_report_file)
    clean_df = clean_data(raw_df)
    processed_df, encoders = encode_features(clean_df, config)
    target_path = output_path or config.processed_data_file
    target_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(target_path, index=False)
    save_artifact(encoders, config.preprocessor_file)
    save_json({"selected_features": config.selected_features, "target_column": config.target_column}, config.predictions_schema_file)
    LOGGER.info("Preprocessed data saved to %s", target_path)
    return processed_df


def get_train_test_data(df: pd.DataFrame | None = None, config: ProjectConfig = CONFIG):
    """Return stratified train/test splits for reproducible model development."""
    data = df if df is not None else preprocess_data(config=config)
    x = data[config.selected_features]
    y = data[config.target_column]
    return train_test_split(x, y, test_size=config.test_size, random_state=config.random_seed, stratify=y)


if __name__ == "__main__":
    preprocess_data()
