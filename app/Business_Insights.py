"""Business insights page for PredictX."""
from __future__ import annotations

import streamlit as st

from src.business_insights import calculate_business_insights, executive_summary


def render() -> None:
    """Render maintenance economics and reliability metrics."""
    st.title("Business Insights")
    col1, col2 = st.columns(2)
    downtime_cost = col1.number_input("Downtime Cost per Hour ($)", min_value=100.0, value=7500.0, step=500.0)
    downtime_hours = col2.number_input("Average Downtime per Failure (hours)", min_value=0.5, value=6.0, step=0.5)
    preventable = st.slider("Preventable Failure Rate with Predictive Maintenance", 0.0, 1.0, 0.35)
    with st.spinner("Calculating business impact..."):
        insights = calculate_business_insights(
            downtime_cost_per_hour=downtime_cost,
            average_downtime_hours=downtime_hours,
            preventable_failure_rate=preventable,
        )
    cols = st.columns(4)
    cols[0].metric("Failure Rate", f"{insights['failure_rate']:.2%}")
    cols[1].metric("Downtime Cost", f"${insights['estimated_downtime_cost']:,.0f}")
    cols[2].metric("Potential Savings", f"${insights['potential_cost_savings']:,.0f}")
    cols[3].metric("High Wear Rate", f"{insights['high_tool_wear_asset_rate']:.2%}")
    st.success(executive_summary(insights))
    st.json(insights)
