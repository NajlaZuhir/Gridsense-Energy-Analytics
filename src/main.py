from pathlib import Path
import pandas as pd

from preprocessing import preprocess_data
from features import final_df
from train import run_train


def run_ml_pipeline(meter_id, df):

    # 1. Select requested meter
    meter_df = df[df["meter_id"] == meter_id].copy()

    if meter_df.empty:
        raise ValueError(f"No records found for meter {meter_id}.")

    # 2. Train, evaluate and save model
    metrics = run_train(meter_df, meter_id=meter_id)

    return metrics


if __name__ == "__main__":

    selected_clients = [
        "T282", "T39", "T106", "T201", "T279",
        "T7", "T248", "T136", "T316", "T317"
    ]

    raw_path = Path("data") / "raw" / "electricity_hourly_dataset.tsf"

    # 1. Preprocess once
    df = preprocess_data(raw_path)

    # 2. Generate features once
    df = final_df(df)

    # 3. Run pipeline for each client
    all_metrics = []

    for meter_id in selected_clients:

        print(f"\nRunning ML pipeline for meter: {meter_id}")

        metrics = run_ml_pipeline(meter_id, df)

        all_metrics.append(metrics)

    # 4. Save evaluation results
    results_df = pd.DataFrame(all_metrics)

    results_path = (
        Path("data") / "results" / "per_client_forecasting_metrics.csv"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(results_path, index=False)

    print("\nPer-client forecasting results:")
    print(results_df.to_string(index=False))

    print(f"\nMetrics saved to {results_path}")