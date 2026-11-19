"""Feature engineering for forecasting and anomaly detection"""
import numpy as np
import os
import pandas as pd

def create_forecast_features(df):
    df = df.copy()

    # Calendar features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Lag features
    consumption_by_meter = df.groupby("meter_id")["consumption_kw"]
    df["lag_24h"] = consumption_by_meter.shift(24)
    df["lag_48h"] = consumption_by_meter.shift(48)
    df["lag_168h"] = consumption_by_meter.shift(168)

    df["target_date"] = df.index.normalize()
    daily_consumption = (
        df.groupby(["meter_id", "target_date"])["consumption_kw"]
        .agg(["mean", "std"])
    )
    previous_day = daily_consumption.groupby(level="meter_id").shift(1)
    previous_day.columns = ["previous_day_mean", "previous_day_std"]
    df = df.join(previous_day, on=["meter_id", "target_date"])

    return df

def create_anomaly_features(df):
    df = df.copy()

    # Rolling statistics
    df["rolling_mean_24h"] = (
        df.groupby("meter_id")["consumption_kw"]
        .transform(lambda values: values.shift(1).rolling(window=24).mean())
    )
    df["rolling_std_24h"] = (
        df.groupby("meter_id")["consumption_kw"]
        .transform(lambda values: values.shift(1).rolling(window=24).std())
    )
    # Hour sine
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )
    # Hour cosine
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Deviation from previous 24-hour mean
    df["deviation_from_mean"] = (
        df["consumption_kw"]
        - df["rolling_mean_24h"]
    )
    return df

def final_df(df):
    df = create_forecast_features(df)
    df = create_anomaly_features(df)

    # target; holds the actual hourly consumption we want the ML model to predict.
    df["target_kw"] = df["consumption_kw"]

    # Remove rows with missing engineered features
    df = df.dropna().copy()

    if df.isnull().sum().sum() ==0:
        final_path = os.path.join("data\\processed", "features_electricity_hourly_dataset.parquet")
        df.to_parquet(final_path, index=True)
    return df

if __name__ == "__main__":

    data_path = os.path.join("data\\processed", "cleaned_electricity_hourly_dataset.parquet")
    processed_data = pd.read_parquet(data_path)
    df = final_df(processed_data)

    df_t1 = df[df["meter_id"] == "T1"].copy()
    t1_path = os.path.join(f"data\\processed", f"T1_electricity_hourly_dataset.parquet")
    df_t1.to_parquet(t1_path, index=True)

    print("Final features dataframe shape:", df.shape)
    print("T1 features dataframe shape:", df_t1.shape)
    print("Final columns:", df.columns)