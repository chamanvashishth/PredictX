"""Utilities for comparing trained PredictX models."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.train import train_models
from src.utils import ProjectConfig, SCREENSHOTS_DIR, load_json

CONFIG = ProjectConfig()


def load_or_train_metrics(config: ProjectConfig = CONFIG) -> Dict[str, object]:
    """Load metrics from disk or run training if metrics are unavailable."""
    metrics = load_json(config.model_metrics_file)
    if metrics:
        return metrics
    _, trained_metrics, best_model = train_models(config=config)
    return {"best_model": best_model, "metrics": trained_metrics}


def metrics_dataframe(config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Return a tidy model metrics table."""
    payload = load_or_train_metrics(config)
    return pd.DataFrame(payload["metrics"]).T.sort_values("f1", ascending=False)


def save_model_comparison_plot(config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Save grouped metric comparison chart."""
    df = metrics_dataframe(config)
    fig, ax = plt.subplots(figsize=(9, 5))
    df[["precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("PredictX Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)
    return df


if __name__ == "__main__":
    save_model_comparison_plot()
