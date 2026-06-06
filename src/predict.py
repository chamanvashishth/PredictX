"""Prediction engine for industrial machine failure risk scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.train import train_models
from src.utils import ProjectConfig, get_logger, load_artifact

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()


@dataclass(frozen=True)
class MachineInput:
    """Validated input fields required for a single machine prediction."""

    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


def risk_category(probability: float) -> str:
    """Map a failure probability to an operational risk band."""
    if probability >= 0.7:
        return "Critical"
    if probability >= 0.4:
        return "High"
    if probability >= 0.2:
        return "Medium"
    return "Low"


def maintenance_action(category: str) -> str:
    """Recommend the next maintenance action for a risk category."""
    return {
        "Critical": "Stop or derate machine, inspect immediately, and schedule urgent corrective maintenance.",
        "High": "Schedule maintenance in the next shift and inspect torque, temperature, and tool wear drivers.",
        "Medium": "Increase monitoring frequency and plan preventive maintenance during the next available window.",
        "Low": "Continue normal operation with routine condition monitoring.",
    }[category]


def _encode_machine_type(machine_type: str, config: ProjectConfig = CONFIG) -> int:
    encoders = load_artifact(config.preprocessor_file)
    encoder = encoders["Type"]
    normalized = machine_type.upper()[0]
    if normalized not in encoder.classes_:
        raise ValueError(f"Unknown machine type '{machine_type}'. Expected one of {list(encoder.classes_)}")
    return int(encoder.transform([normalized])[0])


def _to_dataframe(machine_input: MachineInput, config: ProjectConfig = CONFIG) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": _encode_machine_type(machine_input.machine_type, config),
                "Air temperature [K]": machine_input.air_temperature,
                "Process temperature [K]": machine_input.process_temperature,
                "Rotational speed [rpm]": machine_input.rotational_speed,
                "Torque [Nm]": machine_input.torque,
                "Tool wear [min]": machine_input.tool_wear,
            }
        ],
        columns=config.selected_features,
    )


def predict_failure(machine_input: MachineInput, config: ProjectConfig = CONFIG) -> Dict[str, object]:
    """Predict failure, probability, risk category, and maintenance recommendation."""
    if not config.best_model_file.exists() or not config.preprocessor_file.exists():
        train_models(config=config)
    model = load_artifact(config.best_model_file)
    features = _to_dataframe(machine_input, config)
    probability = float(model.predict_proba(features)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(features)[0])
    category = risk_category(probability)
    return {
        "failure_prediction": int(probability >= 0.5),
        "failure_probability": round(probability, 4),
        "risk_category": category,
        "recommended_action": maintenance_action(category),
    }


if __name__ == "__main__":
    print(predict_failure(MachineInput("L", 300.0, 310.0, 1500, 40.0, 100)))
