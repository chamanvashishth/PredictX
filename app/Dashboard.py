"""Operational dashboard page for PredictX."""
from __future__ import annotations

import streamlit as st

from src.business_insights import calculate_business_insights
from src.data_preprocessing import preprocess_data
from src.eda import generate_eda_assets
from src.utils import ProjectConfig

CONFIG = ProjectConfig()


def render() -> None:
    """Render data quality, KPI, and EDA dashboard views."""
    st.title("Industrial IoT Dashboard")
    with st.spinner("Loading processed sensor data..."):
        data = preprocess_data(config=CONFIG)
        insights = calculate_business_insights(data, config=CONFIG)
    cols = st.columns(4)
    cols[0].metric("Machines", f"{len(data):,}")
    cols[1].metric("Failure Rate", f"{insights['failure_rate']:.2%}")
    cols[2].metric("Failures", f"{insights['failure_count']:,}")
    cols[3].metric("Potential Savings", f"${insights['potential_cost_savings']:,.0f}")
    st.subheader("Sensor Dataset Preview")
    st.dataframe(data.head(50), use_container_width=True)
    if st.button("Generate / Refresh EDA Assets"):
        with st.spinner("Creating EDA charts..."):
            generate_eda_assets(CONFIG)
        st.success("EDA assets saved to screenshots/.")
    st.subheader("Interactive Sensor Distributions")
    feature = st.selectbox("Feature", CONFIG.selected_features[1:])
    st.bar_chart(data[feature].value_counts(bins=20).sort_index())
