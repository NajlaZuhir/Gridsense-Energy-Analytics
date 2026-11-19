
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

feature_cols = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "previous_day_mean",
    "previous_day_std"
]


def forecast_next_24h(model, data, feature_cols):
    """
    Forecast the next 24 hours using completed daily history.
    """

    data = data.copy()

    # Support timestamp as column or index
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")

    data = data.sort_index()

    # Validate data
    if len(data) < 168:
        raise ValueError("At least 168 hourly readings required.")

    if data.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")

    if data["consumption_kw"].isna().any():
        raise ValueError("Missing consumption readings detected.")

    if not data.index.to_series().diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Readings must be consecutive hourly data.")

    if data.index[-1].hour != 23:
        raise ValueError(
            "Forecast requires the final hour (23:00) of a completed day."
        )

    # Last completed day
    yesterday = data["consumption_kw"].iloc[-24:]

    # Generate next 24 hourly timestamps
    future_index = pd.date_range(
        start=data.index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    future = pd.DataFrame(index=future_index)

    # 1. Calendar features
    future["hour"] = future.index.hour
    future["day_of_week"] = future.index.dayofweek
    future["month"] = future.index.month
    future["is_weekend"] = (
        future.index.dayofweek >= 5
    ).astype(int)

    # 2. Lag features
    future["lag_24h"] = data["consumption_kw"].iloc[-24:].values
    future["lag_48h"] = data["consumption_kw"].iloc[-48:-24].values
    future["lag_168h"] = data["consumption_kw"].iloc[-168:-144].values

    # 3. Previous-day statistics
    future["previous_day_mean"] = yesterday.mean()
    future["previous_day_std"] = yesterday.std()

    # 4. Generate forecast
    future["predicted_kw"] = model.predict(
        future[feature_cols]
    )

    return future[["predicted_kw"]]


if __name__ == "__main__":

    # Load historical dataset
    df = pd.read_parquet(
        "data/processed/T1_electricity_hourly_dataset.parquet"
    )

    # Select an historical forecast date for testing
    if "timestamp" not in df.columns:
        df = df.reset_index()

    history = df[
        df["timestamp"] < "2014-08-15"
    ][["timestamp", "consumption_kw"]].copy()

    # Load trained XGBoost model
    model = XGBRegressor()
    model.load_model("models/T1_xgboost.json")

    # Generate forecast
    forecast = forecast_next_24h(
        model=model,
        data=history,
        feature_cols=feature_cols
    )

    print(forecast)
    print("Forecast hours:", len(forecast))
