"""Explainability page for PredictX."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.explain import business_interpretation, generate_shap_explanations
from src.feature_importance import get_feature_importance


def render() -> None:
    """Render model explanation assets and interpretation layer."""
    st.title("Explainable AI")
    if st.button("Generate Explanations"):
        with st.spinner("Generating SHAP-compatible explanation plots..."):
            generate_shap_explanations()
        st.success("Explanation assets saved to screenshots/.")
    importance = get_feature_importance()
    st.subheader("Feature Importance")
    st.dataframe(importance, use_container_width=True)
    st.bar_chart(importance.set_index("feature"))
    st.subheader("Business Interpretation Layer")
    for _, row in importance.head(6).iterrows():
        st.write("• " + business_interpretation(str(row["feature"]), float(row["importance"])))
    for path in [Path("screenshots/shap_global.png"), Path("screenshots/shap_local.png")]:
        if path.exists():
            st.image(str(path), caption=path.name, use_container_width=True)
