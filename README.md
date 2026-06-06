# PredictX: Industrial IoT Predictive Maintenance Platform

PredictX is a production-oriented machine learning repository for predicting industrial equipment failures using the **AI4I 2020 Predictive Maintenance Dataset**. It demonstrates the full applied ML lifecycle expected in a real manufacturing, logistics, or smart-factory reliability program.

![PredictX Architecture](architecture.svg)

## Project Overview

Unexpected machine failures create downtime, maintenance cost, throughput loss, and operational uncertainty. PredictX turns sensor telemetry into actionable risk scores so maintenance teams can prioritize proactive inspections and schedule interventions before failures occur.

The platform includes:

- Data ingestion from the AI4I 2020 dataset with deterministic offline fallback.
- Data validation, duplicate and missing-value checks, quality reporting, and feature encoding.
- EDA visualizations automatically saved to `screenshots/`.
- Reproducible model training for Logistic Regression, Decision Tree, and Random Forest.
- Model evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report.
- Best-model persistence with Joblib.
- Explainability assets with feature importance and SHAP-compatible global/local explanations.
- Business insights for failure rate, downtime exposure, cost savings, and risk metrics.
- A professional multi-page Streamlit dashboard.

## Business Value

PredictX helps maintenance teams answer four core questions:

1. Which machines are most likely to fail?
2. What operational factors are driving risk?
3. How much downtime cost is exposed?
4. What maintenance action should be taken now?

The prediction engine returns a failure probability, risk category, and recommended action so technical model output becomes an operational decision.

## Architecture

The repository follows a modular lifecycle architecture:

1. **IIoT Sensors**: machine type, air temperature, process temperature, rotational speed, torque, and tool wear.
2. **Data Pipeline**: ingestion, validation, missing-value handling, duplicate detection, quality reporting, and export.
3. **Feature Layer**: label encoding, selected features, train/test split, and schema persistence.
4. **Training Layer**: reproducible model registry and hyperparameter configuration.
5. **Evaluation Layer**: metrics, confusion matrix, classification report, and model comparison.
6. **Explainability Layer**: feature importance, SHAP global explanations, SHAP local explanations, and business interpretation.
7. **Application Layer**: Streamlit pages for dashboards, prediction, comparison, explainability, and business insights.

## Dataset Description

PredictX uses the AI4I 2020 Predictive Maintenance Dataset. The target is:

- `Machine failure`

Modeling features are:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The preprocessing module downloads the dataset from UCI when available and creates a deterministic AI4I-like fallback dataset when external network access is unavailable, keeping the project executable in restricted environments.

## Repository Structure

```text
PredictX/
├── app/                         # Streamlit pages
├── src/                         # ML lifecycle source modules
├── data/raw/                    # Raw dataset location
├── data/processed/              # Processed data and quality reports
├── models/                      # Joblib and metric artifacts
├── screenshots/                 # Generated EDA/evaluation/explainability images
├── notebooks/PredictX_EDA.ipynb # EDA notebook
├── tests/                       # Unit tests
├── requirements.txt
├── streamlit_app.py
├── README.md
├── LICENSE
├── .gitignore
└── architecture.svg
```

## EDA Findings to Review

Run the EDA pipeline to create:

- Histograms of sensor measurements.
- Correlation heatmap.
- Failure distribution plot.
- Boxplots by failure status.
- Feature-to-failure correlation analysis.
- Statistical summary CSV.

```bash
python -m src.eda
```

Generated assets are written to `screenshots/`.

## Model Training

The training pipeline compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest

It uses stratified train/test splitting, a fixed random seed, class-weighted models, and reproducible hyperparameters.

```bash
python -m src.train
```

Artifacts:

- `models/best_model.joblib`
- `models/model_metrics.json`
- `models/preprocessor.joblib`
- `models/prediction_schema.json`

## Evaluation Results

Run evaluation after training:

```bash
python -m src.evaluate
```

The evaluation pipeline generates:

- ROC-AUC
- Confusion matrix image
- Classification report
- Structured `models/evaluation_report.json`

## Explainable AI

PredictX includes a layered explainability workflow:

- Native model feature importance.
- SHAP global explanations when the installed environment supports SHAP.
- SHAP/local fallback explanation charts when SHAP execution is unavailable.
- Business interpretation text for each major feature.

```bash
python -m src.explain
```

## Prediction Engine

Single-machine scoring accepts:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

It returns:

- Failure prediction
- Failure probability
- Risk category
- Recommended maintenance action

Example:

```python
from src.predict import MachineInput, predict_failure

result = predict_failure(MachineInput("L", 300.0, 310.0, 1500, 40.0, 120))
print(result)
```

## Streamlit Web Application

Run the dashboard locally:

```bash
streamlit run streamlit_app.py
```

Pages:

1. Home
2. Dashboard
3. Prediction
4. Model Comparison
5. Explainability
6. Business Insights

## Deployment Guide

A typical deployment path is:

1. Install dependencies with `pip install -r requirements.txt`.
2. Run `python -m src.results_summary` to generate data, model, evaluation, EDA, and explainability artifacts.
3. Launch the web app with `streamlit run streamlit_app.py`.
4. Package the app in a container or deploy to Streamlit Community Cloud, AWS App Runner, ECS, or an internal MLOps platform.
5. Schedule retraining using CI/CD, Airflow, Prefect, or a managed ML pipeline.

## Testing

```bash
pytest -q
```

Tests cover:

- Data preprocessing
- Training pipeline
- Prediction pipeline

## Screenshots

After running the pipelines, generated screenshots include:

- `screenshots/histograms.png`
- `screenshots/correlation_heatmap.png`
- `screenshots/failure_distribution.png`
- `screenshots/boxplots.png`
- `screenshots/feature_correlation_analysis.png`
- `screenshots/model_comparison.png`
- `screenshots/feature_importance.png`
- `screenshots/shap_global.png`
- `screenshots/shap_local.png`
- `screenshots/confusion_matrix.png`

## Future Improvements

- Add streaming ingestion from MQTT, Kafka, or AWS IoT Core.
- Add model monitoring for drift, calibration, and data-quality anomalies.
- Integrate a feature store and experiment tracking system.
- Add batch and REST prediction services.
- Extend labels to failure modes and remaining useful life estimation.
- Add CI/CD workflows and Docker deployment manifests.

## License

This project is licensed under the MIT License.
