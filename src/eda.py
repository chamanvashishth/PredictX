"""Exploratory data analysis assets for PredictX."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preprocessing import load_raw_data, preprocess_data
from src.utils import ProjectConfig, SCREENSHOTS_DIR, ensure_directories, get_logger

LOGGER = get_logger(__name__)
CONFIG = ProjectConfig()
sns.set_theme(style="whitegrid")


def generate_eda_assets(config: ProjectConfig = CONFIG) -> Dict[str, str]:
    """Generate required EDA charts and return their file paths."""
    ensure_directories()
    raw = load_raw_data(config=config)
    processed = preprocess_data(config=config)
    paths: Dict[str, str] = {}

    fig = raw[config.selected_features[1:]].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Sensor Feature Histograms", y=1.02)
    histogram_path = SCREENSHOTS_DIR / "histograms.png"
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=180)
    plt.close("all")
    paths["histograms"] = str(histogram_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(processed.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    heatmap_path = SCREENSHOTS_DIR / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    paths["correlation_heatmap"] = str(heatmap_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    raw[config.target_column].map({0: "Healthy", 1: "Failure"}).value_counts().plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Machine Failure Distribution")
    ax.set_ylabel("Count")
    failure_path = SCREENSHOTS_DIR / "failure_distribution.png"
    fig.tight_layout()
    fig.savefig(failure_path, dpi=180)
    plt.close(fig)
    paths["failure_distribution"] = str(failure_path)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.ravel(), config.selected_features[1:]):
        sns.boxplot(data=raw, x=config.target_column, y=column, ax=ax)
        ax.set_xticklabels(["Healthy", "Failure"])
    fig.suptitle("Sensor Distributions by Failure Status")
    boxplot_path = SCREENSHOTS_DIR / "boxplots.png"
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=180)
    plt.close(fig)
    paths["boxplots"] = str(boxplot_path)

    correlations = processed.corr(numeric_only=True)[config.target_column].drop(config.target_column).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    correlations.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Feature Correlation with Machine Failure")
    ax.set_ylabel("Pearson correlation")
    feature_corr_path = SCREENSHOTS_DIR / "feature_correlation_analysis.png"
    fig.tight_layout()
    fig.savefig(feature_corr_path, dpi=180)
    plt.close(fig)
    paths["feature_correlation_analysis"] = str(feature_corr_path)

    summary_path = SCREENSHOTS_DIR / "statistical_summary.csv"
    raw[config.selected_features[1:] + [config.target_column]].describe().round(3).to_csv(summary_path)
    paths["statistical_summary"] = str(summary_path)
    LOGGER.info("EDA assets generated: %s", paths)
    return paths


if __name__ == "__main__":
    generate_eda_assets()
