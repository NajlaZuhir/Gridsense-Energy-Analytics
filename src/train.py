
import os
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_split import forecast_cols, forecast_split as Chronological_data_split



def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    }


def train_xgboost(X_train, y_train):
    # Initialize XGBoost
    xgb_model_client = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # Train on training data only
    xgb_model_client.fit(X_train, y_train)

    return xgb_model_client

def evaluate_xgboost(xgb_model, X_val, y_val):
    predictions = xgb_model.predict(X_val)
    metrics = evaluate_forecast(y_val, predictions)
    print("Validation Metrics:", metrics)
    return metrics


def test_xgboost(xgb_model, X_test, y_test):
    predictions = xgb_model.predict(X_test)
    metrics = evaluate_forecast(y_test, predictions)
    print("Test Metrics:", metrics)
    return metrics


def save_model(xgb_model, meter_id="T1"):
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{meter_id}_xgboost.json"

    xgb_model.save_model(model_path)

    print(f"Model saved to {model_path}")

    print("Model saved successfully.")

def train_and_evaluate(
    df_t,
    feature_cols=forecast_cols,
    target_col="target_kw",
    date_val="2014-01-01",
    date_test="2014-07-01"
):
    # 1. Split chronologically
    X_train, y_train, X_val, y_val, X_test, y_test = (
        Chronological_data_split(
            df_t, feature_cols, target_col,
            date_val, date_test
        )
    )

    # 2. Train on training data
    model = train_xgboost(X_train, y_train)

    # 3. Evaluate on validation data
    val_metrics = evaluate_xgboost(model, X_val, y_val)

    # 4. Combine training + validation
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    # 5. Retrain selected configuration
    final_model = train_xgboost(
        X_train_final,
        y_train_final
    )

    # 6. Final test evaluation
    test_metrics = test_xgboost(
        final_model,
        X_test,
        y_test
    )

    # 7. Save finalized model
    save_model(final_model)

    return final_model, val_metrics, test_metrics


if __name__ == "__main__":
    # Load the processed features dataset
    t1 = Path("data") / "processed" / "T1_electricity_hourly_dataset.parquet"
    df = pd.read_parquet(t1)

    # Train and evaluate the model
    train_and_evaluate(df)
