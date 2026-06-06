"""Feature importance extraction and plotting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.train import train_models
from src.utils import ProjectConfig, SCREENSHOTS_DIR, load_artifact

CONFIG = ProjectConfig()


def get_feature_importance(config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Return model-native feature importance or absolute coefficient importances."""
    if not config.best_model_file.exists():
        train_models(config=config)
    model = load_artifact(config.best_model_file)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        values = [0.0] * len(config.selected_features)
    return pd.DataFrame({"feature": config.selected_features, "importance": values}).sort_values("importance", ascending=False)


def save_feature_importance_plot(config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Save a bar chart of feature importances."""
    df = get_feature_importance(config)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"], color="#004c97")
    ax.invert_yaxis()
    ax.set_title("PredictX Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)
    return df


if __name__ == "__main__":
    save_feature_importance_plot()
