import os
import pandas as pd
import numpy as np
import requests
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta, timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. MLflow setup
mlflow.set_tracking_uri("http://34.58.217.248:5000")
client = MlflowClient()
model_name = "model"
model_uri = f"models:/{model_name}@champion"
model = mlflow.pyfunc.load_model(model_uri)


# 2. Weather fetching
def fetch_weather_2_days_ago():
    target_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime('%Y-%m-%d')
    hourly_vars = [
        'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m', 'apparent_temperature',
        'cloudcover', 'cloudcover_low', 'windspeed_10m', 'winddirection_10m',
        'surface_pressure', 'vapour_pressure_deficit', 'weathercode',
        'wet_bulb_temperature_2m', 'precipitation', 'is_day'
    ]
    params = {
        'latitude': 23.8103,
        'longitude': 90.4125,
        'start_date': target_date,
        'end_date': target_date,
        'hourly': ','.join(hourly_vars),
        'timezone': 'Asia/Dhaka'
    }
    print(f"Fetching weather data for {target_date}...")
    res = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    res.raise_for_status()
    return pd.DataFrame(res.json()["hourly"])


# 3. Feature engineering
def engineer_features(df):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    for lag in range(1, 7):
        df[f'temp_lag_{lag}'] = df['temperature_2m'].shift(lag)

    df['humidity_ewm3'] = df['relative_humidity_2m'].ewm(span=3, adjust=False).mean()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_of_year'] = df['time'].dt.isocalendar().week
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    df = df.drop(columns=['time'])

    y = df['precipitation'] if 'precipitation' in df.columns else None
    X = df.drop(columns=['precipitation']) if y is not None else df

    for col in [col for col in X.columns if 'temp_lag_' in col]:
        X[col] = X[col].fillna(X['temperature_2m'].iloc[0])

    return X, y


# 4. Get logged metrics from MLflow
def get_logged_metrics_from_champion(model_name="model", stage="champion"):
    latest_ver = client.get_latest_versions(model_name, stages=[stage])[0]
    run_id = latest_ver.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics
    print(f"Fetched metrics from run {run_id}: {metrics}")
    return metrics


# 5. Current metrics
def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


# 6. Compare metrics
def compare_metrics(current, logged, thresholds={"mae": 0.2, "mse": 0.2, "r2": 0.05}):
    print("🔍 Checking for drift in metrics:")
    drift_detected = False
    for metric, threshold in thresholds.items():
        if metric not in current or metric not in logged:
            print(f"⚠️ Metric '{metric}' missing.")
            continue
        curr_val = current[metric]
        logged_val = logged[metric]
        diff = abs(curr_val - logged_val) / (abs(logged_val) + 1e-6)
        print(f"{metric.upper()}: Current={curr_val:.4f}, Logged={logged_val:.4f}, Drift={diff:.4f}")
        if diff > threshold:
            print(f"🚨 Drift detected in {metric.upper()}!")
            drift_detected = True
        else:
            print(f"✅ {metric.upper()} OK.")
    return drift_detected


# 7. Run monitoring
def run_monitoring():
    df = fetch_weather_2_days_ago()
    X, y_true = engineer_features(df)
    y_pred = model.predict(X)

    if y_true is None:
        print("⚠️ Cannot calculate metrics – no ground truth available.")
        return

    current_metrics = calculate_metrics(y_true, y_pred)
    logged_metrics = get_logged_metrics_from_champion(model_name)
    drift = compare_metrics(current_metrics, logged_metrics)

    if drift:
        print("🎯 ACTION: Consider triggering retraining.")
    else:
        print("🟢 No drift. Model performance stable.")


if __name__ == "__main__":
    run_monitoring()
