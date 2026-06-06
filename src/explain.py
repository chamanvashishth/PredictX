"""Explainable AI layer for PredictX with SHAP-compatible fallbacks."""
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.feature_importance import get_feature_importance, save_feature_importance_plot
from src.train import train_models
from src.utils import ProjectConfig, SCREENSHOTS_DIR, get_logger, load_artifact

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()


def _load_model_and_data(config: ProjectConfig = CONFIG) -> Tuple[object, pd.DataFrame]:
    if not config.best_model_file.exists():
        train_models(config=config)
    data = preprocess_data(config=config)
    return load_artifact(config.best_model_file), data[config.selected_features]


def generate_shap_explanations(config: ProjectConfig = CONFIG) -> Dict[str, str]:
    """Generate global and local explanation plots using SHAP when available."""
    model, features = _load_model_and_data(config)
    sample = features.sample(min(300, len(features)), random_state=config.random_seed)
    paths = {
        "feature_importance": str(SCREENSHOTS_DIR / "feature_importance.png"),
        "shap_global": str(SCREENSHOTS_DIR / "shap_global.png"),
        "shap_local": str(SCREENSHOTS_DIR / "shap_local.png"),
    }
    save_feature_importance_plot(config)
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.Explainer(model.predict_proba, sample)
        shap_values = explainer(sample)
        values = shap_values.values[:, :, 1] if getattr(shap_values, "values", np.array([])).ndim == 3 else shap_values.values
        shap.summary_plot(values, sample, show=False)
        plt.tight_layout()
        plt.savefig(paths["shap_global"], dpi=180, bbox_inches="tight")
        plt.close()
        first = values[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        pd.Series(first, index=sample.columns).sort_values().plot(kind="barh", ax=ax, color="#ff7f0e")
        ax.set_title("Local SHAP Contributions for One Machine")
        fig.tight_layout()
        fig.savefig(paths["shap_local"], dpi=180)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("SHAP failed (%s). Using feature-importance fallback plots.", exc)
        importance = get_feature_importance(config)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance["feature"], importance["importance"], color="#9467bd")
        ax.invert_yaxis()
        ax.set_title("Global Explanation Fallback: Feature Importance")
        fig.tight_layout()
        fig.savefig(paths["shap_global"], dpi=180)
        plt.close(fig)
        row = sample.iloc[0]
        contribution = row.rank(pct=True) * importance.set_index("feature").reindex(row.index)["importance"].fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4))
        contribution.sort_values().plot(kind="barh", ax=ax, color="#ff7f0e")
        ax.set_title("Local Explanation Fallback: Ranked Risk Drivers")
        fig.tight_layout()
        fig.savefig(paths["shap_local"], dpi=180)
        plt.close(fig)
    return paths


def business_interpretation(feature: str, value: float) -> str:
    """Translate feature effects into maintenance-oriented language."""
    rules = {
        "Tool wear [min]": "High tool wear indicates consumable degradation and should trigger tool inspection or replacement.",
        "Torque [Nm]": "Elevated torque can indicate friction, overload, or mechanical binding.",
        "Rotational speed [rpm]": "Abnormal speed patterns may reveal motor or process-control instability.",
        "Air temperature [K]": "High ambient temperature increases thermal stress on machine components.",
        "Process temperature [K]": "High process temperature can reduce equipment reliability and lubricant effectiveness.",
        "Type": "Machine type captures asset class differences that may require tailored maintenance thresholds.",
    }
    return f"{feature}={value}: {rules.get(feature, 'Monitor this factor as a potential reliability signal.')}"


if __name__ == "__main__":
    generate_shap_explanations()
