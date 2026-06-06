"""Evaluation reports and diagnostic plots for PredictX models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score

from src.data_preprocessing import get_train_test_data, preprocess_data
from src.train import train_models
from src.utils import ProjectConfig, SCREENSHOTS_DIR, get_logger, load_artifact, save_json

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()


def evaluate_best_model(config: ProjectConfig = CONFIG) -> Dict[str, Any]:
    """Evaluate the persisted best model and save structured diagnostics."""
    if not config.best_model_file.exists():
        train_models(config=config)
    data = preprocess_data(config=config)
    _, x_test, _, y_test = get_train_test_data(data, config)
    model = load_artifact(config.best_model_file)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    save_json(results, Path("models/evaluation_report.json"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["Healthy", "Failure"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Best Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return results


if __name__ == "__main__":
    evaluate_best_model()
