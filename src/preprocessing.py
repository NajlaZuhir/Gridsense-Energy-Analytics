"""Data loading, cleaning, validation"""
import pandas as pd
import os

def load_data(file_path):
  rows = []

  with open(file_path, "r", encoding="utf-8") as file:
      data_started = False

      for line in file:
          line = line.strip()

          if line == "@data":
              data_started = True
              continue

          if not data_started or not line:
              continue

          series_name, start_time, values = line.split(":", 2)

          start_timestamp = pd.to_datetime(
              start_time,
              format="%Y-%m-%d %H-%M-%S"
          )

          for hour, value in enumerate(values.split(",")):
              rows.append({
                  "timestamp": start_timestamp + pd.to_timedelta(hour, unit="h"),
                  "meter_id": series_name,
                  "consumption_kw": float(value)
              })
  return rows


def clean_data(data):
    """Ensures correct data types in the dataset."""
    df_clean = data.copy()
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    df_clean["meter_id"] = df_clean["meter_id"].astype("string")
    df_clean["consumption_kw"] = pd.to_numeric(
        df_clean["consumption_kw"], errors="coerce"
    )
    return df_clean

def validate_data(data, duplicates_count, irregular_intervals_series):
    """Validates that the dataset meets required quality standards."""
    missing_check = data[["timestamp", "meter_id", "consumption_kw"]].isnull().sum().sum() == 0
    duplicates_check = duplicates_count == 0
    negative_check = (data["consumption_kw"] < 0).sum() == 0
    irregular_check = irregular_intervals_series.sum() == 0
    
    return missing_check and duplicates_check and negative_check and irregular_check


def preprocess_data(file_path):
    """Loads, cleans, and validates the dataset."""
    raw_data = load_data(file_path)
    df_cleaned = clean_data(pd.DataFrame(raw_data))
    
    duplicates_count = df_cleaned.duplicated(subset=["meter_id", "timestamp"]).sum()
    df_cleaned = df_cleaned.sort_values(by=["meter_id", "timestamp"]).reset_index(drop=True)
    
    time_diff = df_cleaned.groupby("meter_id")["timestamp"].diff()
    irregular_intervals_series = time_diff.notna() & (
        time_diff != pd.to_timedelta(1, unit="h")
    )

    validation_passed = validate_data(
        df_cleaned, duplicates_count, irregular_intervals_series
    )

    df_cleaned = df_cleaned.sort_values(["meter_id", "timestamp"])
    df_cleaned = df_cleaned.set_index("timestamp")
    
    if validation_passed:
        processed_path = os.path.join("data\\processed", "cleaned_electricity_hourly_dataset.parquet")
        df_cleaned.to_parquet(processed_path, index=True)
    
    return df_cleaned, validation_passed

if __name__ == "__main__":
    data_path = os.path.join("data\\raw", "electricity_hourly_dataset.tsf")
    cleaned_data, is_valid = preprocess_data(data_path)
    print(f"Data validation passed: {is_valid}")
    print(cleaned_data.head())