"""Model comparison page for PredictX."""
from __future__ import annotations

import streamlit as st

from src.model_comparison import metrics_dataframe, save_model_comparison_plot
from src.train import train_models


def render() -> None:
    """Render model metrics and retraining controls."""
    st.title("Model Comparison")
    if st.button("Retrain Models"):
        with st.spinner("Training candidate models..."):
            _, _, best = train_models()
        st.success(f"Training complete. Best model: {best}")
    try:
        df = metrics_dataframe()
    except Exception:
        with st.spinner("No metrics found. Training models..."):
            train_models()
            df = metrics_dataframe()
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(df[["precision", "recall", "f1", "roc_auc"]])
    if st.button("Save Comparison Plot"):
        save_model_comparison_plot()
        st.success("Saved screenshots/model_comparison.png")
