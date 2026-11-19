from data_split import anomaly_split
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pandas as pd

def train_isolation_forest(anomaly_train):
    # Initialize Isolation Forest
    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )

    # Train using historical data (2012–2013)
    isolation_model.fit(anomaly_train)
    print("Isolation Forest training completed!")

    return isolation_model


def predict_anomalies(isolation_model, anomaly_test):
    anomaly_test = anomaly_test.copy()

    # Predict anomalies on unseen 2014 data
    anomaly_test["anomaly_label"] = (
        isolation_model.predict(anomaly_test)
    )

    # Calculate anomaly scores
    anomaly_test["anomaly_score"] = (
        isolation_model.decision_function(anomaly_test.drop(
            columns=["anomaly_label", "anomaly_score"], errors="ignore"
        ))
    )

    # Convert labels into 0/1
    # 0 = Normal, 1 = Anomaly
    anomaly_test["is_anomaly"] = (
        anomaly_test["anomaly_label"] == -1
    ).astype(int)

    return anomaly_test

def detect_anomalies(df):
    # Split the data for anomaly detection
    anomaly_train, anomaly_test = anomaly_split(df)

    # Train Isolation Forest
    isolation_model = train_isolation_forest(anomaly_train)

    # Predict anomalies on the test set
    anomaly_results = predict_anomalies(isolation_model, anomaly_test)

    # Group and calculate statistics for anomalies and non-anomalies
    anomaly_stats = anomaly_results.groupby("is_anomaly")["anomaly_score"].agg(
        ["count", "min", "max"]
    )
    anomaly_stats.index = anomaly_stats.index.map({
        0: "Non-Anomaly (0)",
        1: "Anomaly (1)"
    })

    print("Anomaly vs Non-Anomaly Summary:")
    print(anomaly_stats)

    return anomaly_results
if __name__ == "__main__":
    # Load the processed features dataset
    t1 = Path("data") / "processed" / "T1_electricity_hourly_dataset.parquet"
    df = pd.read_parquet(t1)

    # Detect anomalies
    anomaly_results = detect_anomalies(df)

    # Save the results to a CSV file
    anomaly_results_path = Path("data") / "results" / "anomaly_results.csv"
    anomaly_results_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_results.to_csv(anomaly_results_path, index=True)
    print(f"Anomaly detection results saved to {anomaly_results_path}")