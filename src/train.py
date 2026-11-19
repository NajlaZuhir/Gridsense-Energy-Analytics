
import os
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from data_split import forecast_cols, forecast_split as Chronological_data_split
from evaluate import evaluate_forecast


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

def save_model(xgb_model, meter_id="T1"):
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{meter_id}_xgboost.json"

    xgb_model.save_model(model_path)

    print(f"Model saved to {model_path}")


def run_train(
    df_t,
    feature_cols=forecast_cols,
    target_col="target_kw",
    date_val="2014-01-01",
    date_test="2014-07-01",
    meter_id="T1"
):
    # 1. Split chronologically
    X_train, y_train, X_val, y_val, X_test, y_test = (
        Chronological_data_split(
            df_t,
            feature_cols,
            target_col,
            date_val,
            date_test
        )
    )

    # 2. Train using training data only
    validation_model = train_xgboost(X_train, y_train)

    # 3. Evaluate on validation data
    val_predictions = validation_model.predict(X_val)
    val_metrics = evaluate_forecast(y_val, val_predictions)

    # 4. Combine training + validation data
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    # 5. Train final model
    final_model = train_xgboost(
        X_train_final,
        y_train_final
    )

    # 6. Evaluate final model on test data
    test_predictions = final_model.predict(X_test)
    test_metrics = evaluate_forecast(y_test, test_predictions)

    # 7. Save final model
    save_model(final_model, meter_id=meter_id)

    print(f"{meter_id} Validation Metrics: {val_metrics}")
    print(f"{meter_id} Test Metrics: {test_metrics}")

    # 8. Return metrics
    return {
        "meter_id": meter_id,
        "val_MAE": val_metrics["MAE"],
        "val_RMSE": val_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"]
    }


if __name__ == "__main__":
    # Load the processed features dataset
    t1 = Path("data") / "processed" / "T1_electricity_hourly_dataset.parquet"
    df = pd.read_parquet(t1)

    # Train and evaluate the model
    run_train(df)