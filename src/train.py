"""Model training and best-model persistence for PredictX."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import get_train_test_data, preprocess_data
from src.utils import ProjectConfig, ensure_directories, get_logger, save_artifact, save_json

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()


def get_model_registry(random_seed: int = CONFIG.random_seed) -> Dict[str, object]:
    """Return configured candidate models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=random_seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def _positive_probability(model: object, x_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    return model.predict(x_test)


def train_models(df: pd.DataFrame | None = None, config: ProjectConfig = CONFIG) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]], str]:
    """Train all candidate models and select the best by F1 then ROC-AUC."""
    ensure_directories()
    data = df if df is not None else preprocess_data(config=config)
    x_train, x_test, y_train, y_test = get_train_test_data(data, config)
    trained_models: Dict[str, object] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for name, model in get_model_registry(config.random_seed).items():
        LOGGER.info("Training %s", name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_score = _positive_probability(model, x_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
        }
        trained_models[name] = model
    best_model_name = max(metrics, key=lambda item: (metrics[item]["f1"], metrics[item]["roc_auc"]))
    save_artifact(trained_models[best_model_name], config.best_model_file)
    save_json({"best_model": best_model_name, "metrics": metrics}, config.model_metrics_file)
    LOGGER.info("Best model persisted: %s", best_model_name)
    return trained_models, metrics, best_model_name


if __name__ == "__main__":
    train_models()
