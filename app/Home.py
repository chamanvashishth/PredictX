"""Home page for the PredictX Streamlit application."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render product overview and operating guidance."""
    st.title("PredictX: Industrial IoT Predictive Maintenance Platform")
    st.markdown(
        """
        PredictX is an end-to-end machine learning platform for predicting industrial equipment failures from
        sensor telemetry. It helps maintenance teams move from reactive repairs to proactive reliability operations.
        """
    )
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Lifecycle", "End-to-End ML")
    kpi_cols[1].metric("Dataset", "AI4I 2020")
    kpi_cols[2].metric("Models", "3 Compared")
    kpi_cols[3].metric("Outputs", "Risk + Action")
    st.subheader("Platform Capabilities")
    st.markdown(
        """
        - Data ingestion, validation, cleaning, and quality reporting
        - Exploratory analysis with manufacturing reliability visualizations
        - Logistic Regression, Decision Tree, and Random Forest model comparison
        - Explainability via feature importance and SHAP-compatible plots
        - Business-impact metrics for downtime exposure and savings
        - Real-time single-machine risk scoring and maintenance recommendations
        """
    )
    st.info("Use the sidebar to navigate between dashboard, prediction, comparison, explainability, and business insights pages.")
