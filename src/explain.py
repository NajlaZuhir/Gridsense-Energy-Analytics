import shap
import pandas as pd
from data_split import forecast_cols
from pathlib import Path
from xgboost import XGBRegressor
import numpy as np



def explain_forecast(model, features = forecast_cols):
    """
    Generate SHAP explanations for electricity load forecasts.

    Parameters:
        model: Trained XGBoost model
        features: DataFrame containing the 9 forecasting features

    Returns:
        DataFrame with predictions and SHAP contributions
    """

    # Initialize SHAP
    explainer = shap.TreeExplainer(model)

    # Calculate predictions
    predictions = model.predict(features)

    # Calculate SHAP values
    shap_values = explainer(features)

    # Create results
    results = pd.DataFrame(
        shap_values.values,
        index=features.index,
        columns=[f"shap_{col}" for col in features.columns]
    )

    results["predicted_kw"] = predictions
    results["base_value_kw"] = shap_values.base_values

    # Reconstruct predictions using SHAP values
    reconstructed = (
        results["base_value_kw"]
        + results[[f"shap_{col}" for col in features.columns]].sum(axis=1)
    )

    # Verify
    if not np.allclose(
        reconstructed,
        results["predicted_kw"],
        rtol=1e-4,
        atol=1e-3
    ):
        raise ValueError("SHAP reconstruction check failed!")

    print("SHAP reconstruction check passed!")

    return results

if __name__ == "__main__":
    data_path = Path("data") / "processed" / "T1_electricity_hourly_dataset.parquet"
    df = pd.read_parquet(data_path)

    model_path = Path("models") / "T1_xgboost.json"
    xgb_model = XGBRegressor()
    xgb_model.load_model(model_path)

    # Generate SHAP explanations
    shap_results = explain_forecast(xgb_model)
    print("SHAP explanations generated successfully.")
    print(shap_results.head())

    # Save SHAP results to a CSV file
    shap_results_path = Path("data") / "results" / "shap_explanations.csv"
    shap_results_path.parent.mkdir(parents=True, exist_ok=True)
    shap_results.to_csv(shap_results_path, index=True)
    print(f"SHAP explanations saved to {shap_results_path}")