"""Streamlit entry point for PredictX."""
from __future__ import annotations

import streamlit as st

from app import Business_Insights, Dashboard, Explainability, Home, Model_Comparison, Prediction

PAGES = {
    "Home": Home.render,
    "Dashboard": Dashboard.render,
    "Prediction": Prediction.render,
    "Model Comparison": Model_Comparison.render,
    "Explainability": Explainability.render,
    "Business Insights": Business_Insights.render,
}


def main() -> None:
    """Configure and run the multi-page Streamlit application."""
    st.set_page_config(page_title="PredictX", page_icon="🏭", layout="wide")
    st.sidebar.title("PredictX Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.caption("Industrial IoT predictive maintenance platform")
    PAGES[page]()


if __name__ == "__main__":
    main()
