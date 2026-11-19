"""
The objective is to answer one question:
Can one model trained on multiple electricity meters forecast consumption as accurately as separate models trained for each meter?
"""


from pathlib import Path
import json

import numpy as np
import pandas as pd

from preprocessing import preprocess_data
from features import final_df
from train import train_xgboost
from evaluate import evaluate_forecast


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SELECTED_CLIENTS = [
    "T282", "T39", "T106", "T201", "T279",
    "T7", "T248", "T136", "T316", "T317"
]

RAW_PATH = Path("data/raw/electricity_hourly_dataset.tsf")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("data/results")

DATE_VAL = "2014-01-01"
DATE_TEST = "2014-07-01"

TARGET_COL = "target_kw"


# --------------------------------------------------
# 1. Prepare global dataset
# --------------------------------------------------

def prepare_global_dataset(df, feature_cols):

    # Ensure timestamp is a column
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            raise ValueError("Timestamp column or index not found.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Keep only selected clients
    df = df[df["meter_id"].isin(SELECTED_CLIENTS)].copy()

    # Sort each client's observations chronologically
    df = df.sort_values(["meter_id", "timestamp"])

    # Check required columns
    required_cols = feature_cols + [TARGET_COL, "meter_id", "timestamp"]

    missing_cols = [
        col for col in required_cols if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Remove rows with missing model inputs or target
    df = df.dropna(subset=required_cols).copy()

    # One-hot encode client identity
    meter_dummies = pd.get_dummies(
        df["meter_id"],
        prefix="meter",
        dtype=int
    )

    # Ensure all clients have a consistent dummy column
    dummy_cols = [
        f"meter_{meter_id}" for meter_id in SELECTED_CLIENTS
    ]

    meter_dummies = meter_dummies.reindex(
        columns=dummy_cols,
        fill_value=0
    )

    # Combine forecasting features and client identity
    X = pd.concat(
        [
            df[feature_cols].reset_index(drop=True),
            meter_dummies.reset_index(drop=True)
        ],
        axis=1
    )

    y = df[TARGET_COL].reset_index(drop=True)

    # Keep timestamps and client IDs for splitting/evaluation
    metadata = df[["timestamp", "meter_id"]].reset_index(drop=True)

    return X, y, metadata


# --------------------------------------------------
# 2. Chronological global split
# --------------------------------------------------

def global_data_split(X, y, metadata):

    train_mask = metadata["timestamp"] < DATE_VAL

    val_mask = (
        (metadata["timestamp"] >= DATE_VAL)
        & (metadata["timestamp"] < DATE_TEST)
    )

    test_mask = metadata["timestamp"] >= DATE_TEST

    return (
        X.loc[train_mask].reset_index(drop=True),
        y.loc[train_mask].reset_index(drop=True),

        X.loc[val_mask].reset_index(drop=True),
        y.loc[val_mask].reset_index(drop=True),

        X.loc[test_mask].reset_index(drop=True),
        y.loc[test_mask].reset_index(drop=True),

        metadata.loc[train_mask].reset_index(drop=True),
        metadata.loc[val_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True)
    )


# --------------------------------------------------
# 3. Calculate metrics
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    metrics = evaluate_forecast(y_true, y_pred)

    wape = (
        np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
        / np.sum(np.abs(np.asarray(y_true)))
    ) * 100

    metrics["WAPE"] = round(wape, 2)

    return metrics


def evaluate_per_client(y_true, y_pred, metadata, period):

    results = []

    for meter_id in SELECTED_CLIENTS:

        mask = metadata["meter_id"] == meter_id

        client_metrics = calculate_metrics(
            y_true.loc[mask],
            y_pred[mask.to_numpy()]
        )

        results.append({
            "meter_id": meter_id,
            "period": period,
            **client_metrics
        })

    return results


# --------------------------------------------------
# 4. Run global training pipeline
# --------------------------------------------------

def run_global_pipeline(df, feature_cols):

    print("\nPreparing global dataset...")

    X, y, metadata = prepare_global_dataset(
        df,
        feature_cols
    )

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        meta_train, meta_val, meta_test
    ) = global_data_split(X, y, metadata)

    print("\nGlobal dataset shapes:")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # ----------------------------------------------
    # Train on training data only
    # ----------------------------------------------

    print("\nTraining validation model...")

    validation_model = train_xgboost(
        X_train,
        y_train
    )

    val_predictions = validation_model.predict(X_val)

    # ----------------------------------------------
    # Retrain on training + validation
    # ----------------------------------------------

    print("\nTraining final global model...")

    X_train_final = pd.concat(
        [X_train, X_val],
        ignore_index=True
    )

    y_train_final = pd.concat(
        [y_train, y_val],
        ignore_index=True
    )

    final_model = train_xgboost(
        X_train_final,
        y_train_final
    )

    # ----------------------------------------------
    # Evaluate final model on test data
    # ----------------------------------------------

    test_predictions = final_model.predict(X_test)

    # Per-client metrics
    val_results = evaluate_per_client(
        y_val,
        val_predictions,
        meta_val,
        "validation"
    )

    test_results = evaluate_per_client(
        y_test,
        test_predictions,
        meta_test,
        "test"
    )

    # Combine results
    client_results = pd.DataFrame(
        val_results + test_results
    )

    # ----------------------------------------------
    # Overall metrics
    # ----------------------------------------------

    overall_results = []

    for period, actual, predicted in [
        ("validation", y_val, val_predictions),
        ("test", y_test, test_predictions)
    ]:

        overall_metrics = calculate_metrics(
            actual,
            predicted
        )

        overall_results.append({
            "period": period,
            "pooled_MAE": overall_metrics["MAE"],
            "pooled_RMSE": overall_metrics["RMSE"],
            "pooled_WAPE": overall_metrics["WAPE"],
            "macro_MAE": round(
                client_results[
                    client_results["period"] == period
                ]["MAE"].mean(), 3
            ),
            "macro_RMSE": round(
                client_results[
                    client_results["period"] == period
                ]["RMSE"].mean(), 3
            ),
            "macro_WAPE": round(
                client_results[
                    client_results["period"] == period
                ]["WAPE"].mean(), 2
            )
        })

    overall_df = pd.DataFrame(overall_results)

    # ----------------------------------------------
    # Save model and feature metadata
    # ----------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "global_xgboost.json"
    final_model.save_model(str(model_path))

    feature_path = MODEL_DIR / "global_features.json"

    with open(feature_path, "w") as f:
        json.dump(list(X.columns), f, indent=4)

    # ----------------------------------------------
    # Save evaluation results
    # ----------------------------------------------

    client_results_path = (
        RESULTS_DIR / "global_forecasting_metrics.csv"
    )

    overall_results_path = (
        RESULTS_DIR / "global_overall_metrics.csv"
    )

    client_results.to_csv(
        client_results_path,
        index=False
    )

    overall_df.to_csv(
        overall_results_path,
        index=False
    )

    print("\nPer-client global model results:")
    print(client_results.to_string(index=False))

    print("\nOverall global model results:")
    print(overall_df.to_string(index=False))

    print(f"\nModel saved to: {model_path}")
    print(f"Client metrics saved to: {client_results_path}")
    print(f"Overall metrics saved to: {overall_results_path}")

    return client_results, overall_df


# --------------------------------------------------
# 5. Execute
# --------------------------------------------------

if __name__ == "__main__":

    print("Preprocessing electricity dataset...")

    df = preprocess_data(RAW_PATH)

    print("Generating forecasting features...")

    df = final_df(df)

    # Import your existing forecasting feature list
    from data_split import forecast_cols

    run_global_pipeline(
        df,
        feature_cols=forecast_cols
    )
