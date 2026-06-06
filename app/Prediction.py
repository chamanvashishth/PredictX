"""Real-time prediction page for PredictX."""
from __future__ import annotations

import streamlit as st

from src.predict import MachineInput, predict_failure


def render() -> None:
    """Render machine failure prediction form."""
    st.title("Failure Risk Prediction")
    with st.form("prediction_form"):
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        col1, col2 = st.columns(2)
        air_temperature = col1.number_input("Air Temperature [K]", 285.0, 320.0, 300.0)
        process_temperature = col2.number_input("Process Temperature [K]", 295.0, 330.0, 310.0)
        rotational_speed = col1.number_input("Rotational Speed [rpm]", 500.0, 3500.0, 1500.0)
        torque = col2.number_input("Torque [Nm]", 1.0, 100.0, 40.0)
        tool_wear = col1.number_input("Tool Wear [min]", 0.0, 300.0, 120.0)
        submitted = st.form_submit_button("Predict Failure Risk")
    if submitted:
        try:
            with st.spinner("Scoring machine risk..."):
                result = predict_failure(
                    MachineInput(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)
                )
            st.metric("Failure Probability", f"{result['failure_probability']:.1%}")
            st.metric("Risk Category", str(result["risk_category"]))
            if result["failure_prediction"]:
                st.error("Failure predicted. Proactive intervention recommended.")
            else:
                st.success("No immediate failure predicted.")
            st.info(str(result["recommended_action"]))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
