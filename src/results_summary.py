"""End-to-end artifact generation summary for PredictX."""
from __future__ import annotations

from typing import Dict

from src.business_insights import calculate_business_insights
from src.eda import generate_eda_assets
from src.evaluate import evaluate_best_model
from src.explain import generate_shap_explanations
from src.model_comparison import save_model_comparison_plot
from src.train import train_models


def run_full_pipeline() -> Dict[str, object]:
    """Execute preprocessing, training, evaluation, explainability, and business reporting."""
    _, metrics, best_model = train_models()
    evaluation = evaluate_best_model()
    eda_assets = generate_eda_assets()
    comparison = save_model_comparison_plot()
    explanations = generate_shap_explanations()
    business = calculate_business_insights()
    return {
        "best_model": best_model,
        "metrics": metrics,
        "evaluation": evaluation,
        "eda_assets": eda_assets,
        "model_comparison_rows": len(comparison),
        "explanations": explanations,
        "business_insights": business,
    }


if __name__ == "__main__":
    print(run_full_pipeline())
