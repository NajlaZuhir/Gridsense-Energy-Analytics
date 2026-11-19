# Select input features
forecast_cols = [
    "hour", "day_of_week", "month", "is_weekend",
    "lag_24h", "lag_48h", "lag_168h", "previous_day_mean",
    "previous_day_std"
]
anomaly_features = [
    "consumption_kw", "hour_sin", "hour_cos", "is_weekend",
    "lag_24h", "rolling_mean_24h", "rolling_std_24h", "deviation_from_mean"
]

def forecast_split(df_t, feature_cols=forecast_cols, target_col="target_kw", date_val="2014-01-01", date_test="2014-07-01"):

    # All data before January 1, 2014. This is used to train the ML models.
    train = df_t[df_t.index < date_val].copy()

    # Data from January 1, 2014, to July 1, 2014 (first half of 2014). This is used to evaluate and tune hyperparameters during development.
    val = df_t[(df_t.index >= date_val) & (df_t.index < date_test)].copy()

    # Data from July 1, 2014, onwards. This is reserved strictly for the final evaluation of the model.
    test = df_t[df_t.index >= date_test].copy()

    # Prepare X and y
    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("The chronological split produced an empty dataset.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def anomaly_split(df, anomaly_features=anomaly_features):
    # Create a separate anomaly detection dataset
    anomaly_df = df[anomaly_features].copy()

    # Chronological split
    anomaly_train = anomaly_df[
        anomaly_df.index < "2014-01-01"
    ].copy()

    # Data from January 1, 2014, onwards. This is used for anomaly detection.
    anomaly_test = anomaly_df[
        anomaly_df.index >= "2014-01-01"
    ].copy()

    print("Training shape:", anomaly_train.shape)
    print("Testing shape:", anomaly_test.shape)
    return anomaly_train, anomaly_test