"""Business value calculations for predictive maintenance decisions."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.data_preprocessing import preprocess_data
from src.utils import ProjectConfig

CONFIG = ProjectConfig()


def calculate_business_insights(
    data: pd.DataFrame | None = None,
    downtime_cost_per_hour: float = 7500.0,
    average_downtime_hours: float = 6.0,
    preventable_failure_rate: float = 0.35,
    config: ProjectConfig = CONFIG,
) -> Dict[str, float]:
    """Estimate failure rate, downtime exposure, savings, and maintenance risk metrics."""
    df = data if data is not None else preprocess_data(config=config)
    failures = float(df[config.target_column].sum())
    total = float(len(df))
    failure_rate = failures / total if total else 0.0
    downtime_cost = failures * average_downtime_hours * downtime_cost_per_hour
    savings = downtime_cost * preventable_failure_rate
    high_wear_rate = float((df["Tool wear [min]"] >= df["Tool wear [min]"].quantile(0.75)).mean())
    torque_alert_rate = float((df["Torque [Nm]"] >= df["Torque [Nm]"].quantile(0.9)).mean())
    return {
        "failure_rate": round(failure_rate, 4),
        "failure_count": int(failures),
        "estimated_downtime_cost": round(downtime_cost, 2),
        "potential_cost_savings": round(savings, 2),
        "high_tool_wear_asset_rate": round(high_wear_rate, 4),
        "torque_alert_rate": round(torque_alert_rate, 4),
    }


def executive_summary(insights: Dict[str, float]) -> str:
    """Create a concise executive summary for operations leaders."""
    return (
        f"Observed failure rate is {insights['failure_rate']:.1%}. "
        f"Estimated downtime exposure is ${insights['estimated_downtime_cost']:,.0f}, "
        f"with potential avoidable savings of ${insights['potential_cost_savings']:,.0f} through proactive maintenance."
    )


if __name__ == "__main__":
    print(calculate_business_insights())
