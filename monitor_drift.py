# monitor_drift.py
# Update Prometheus metric reporting to use labels so can query stuff like:
# model_mae{model_version="v42", city="Dhaka"} 0.1234
# Or alert on drift like:
# model_mae{city="Dhaka"} > 0.25

import os
import pandas as pd
import numpy as np
import requests
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta, timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prefect import flow, task, get_run_logger
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# === Setup ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# === Tasks ===

@task
def fetch_weather_2_days_ago():
    logger = get_run_logger()
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
    logger.info(f"📡 Fetching weather data for {target_date}...")
    res = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    res.raise_for_status()
    return pd.DataFrame(res.json()["hourly"])

@task
def inspect_data_for_nans(df):
    logger = get_run_logger()
    logger.info("Inspecting data for missing values...")

    null_counts = df.isnull().sum()
    total_rows = len(df)

    if null_counts.sum() == 0:
        logger.info("No missing values detected in the dataset. Data looks clean!")
    else:
        logger.warning("Missing values detected!")
        logger.warning("Column-wise Null Summary:")

        for col, nulls in null_counts.items():
            if nulls > 0:
                percent = (nulls / total_rows) * 100
                logger.warning(f"  - {col}: {nulls} nulls ({percent:.2f}%)")

        logger.info("Consider handling these before prediction (drop, fill, or flag).")


@task
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

@task
def load_champion_model(model_name="dhaka_city_precipitation_xgb"):
    logger = get_run_logger()
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=[])
    if not latest_versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    
    latest = sorted(latest_versions, key=lambda mv: int(mv.version))[-1]
    model_uri = f"models:/{model_name}/{latest.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Loaded model version {latest.version} from {model_uri}")
    return model, latest.version  # returning model version also


@task
def get_champion_metrics(model_name="dhaka_city_precipitation_xgb"):
    logger = get_run_logger()
    latest_ver = sorted(client.get_latest_versions(model_name, stages=[]), key=lambda mv: int(mv.version))[-1]
    run_id = latest_ver.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics
    logger.info(f"Champion (latest) model metrics from run {run_id}: {metrics}")
    return metrics

@task
def calculate_metrics(y_true, y_pred, model_version, city="Dhaka"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    registry = CollectorRegistry()

    g_mae = Gauge('model_mae', 'Mean Absolute Error of the model', ['model_version', 'city'], registry=registry)
    g_mse = Gauge('model_mse', 'Mean Squared Error of the model', ['model_version', 'city'], registry=registry)
    g_r2  = Gauge('model_r2', 'R2 Score of the model', ['model_version', 'city'], registry=registry)

    # Set with labels
    g_mae.labels(model_version=str(model_version), city=city).set(mae)
    g_mse.labels(model_version=str(model_version), city=city).set(mse)
    g_r2.labels(model_version=str(model_version), city=city).set(r2)

    # Push to gateway
    push_to_gateway('127.0.0.1:9091', job='model_monitoring', registry=registry)

    return {"mae": mae, "mse": mse, "r2": r2}

@task
def compare_metrics(current, logged, thresholds={"mae": 0.01, "mse": 0.01, "r2": 0.05}):
    logger = get_run_logger()
    logger.info("Comparing metrics...")
    drift_detected = False

    for metric, threshold in thresholds.items():
        if metric not in current or metric not in logged:
            logger.warning(f"{metric} not found in current or logged metrics.")
            continue

        curr_val = current[metric]
        logged_val = logged[metric]

        logger.info(f"{metric.upper()} - Logged: {logged_val:.4f}, Current: {curr_val:.4f}")

        if metric in ["mae", "mse"]:
            drift = curr_val - logged_val  # positive = worse
            if drift > threshold:
                logger.warning(f"Drift detected in {metric.upper()}! Increased by {drift:.4f}")
                drift_detected = True
            else:
                logger.info(f"{metric.upper()} is within threshold or improved (Δ={drift:.4f})")

        elif metric == "r2":
            drift = logged_val - curr_val  # positive = worse
            if drift > threshold:
                logger.warning(f"Drift detected in R²! Dropped by {drift:.4f}")
                drift_detected = True
            else:
                logger.info(f"R² is within threshold or improved (Δ={drift:.4f})")

    return drift_detected


# === Flow ===

@flow(name="drift_monitoring_flow")
def drift_monitoring_flow():
    logger = get_run_logger()
    df = fetch_weather_2_days_ago()
    inspect_data_for_nans(df)
    X, y = engineer_features(df)
    model, model_version = load_champion_model()
    y_pred = model.predict(X)

    current_metrics = calculate_metrics(y, y_pred, model_version=model_version, city="Dhaka")
    champion_metrics = get_champion_metrics()

    drift_detected = compare_metrics(current_metrics, champion_metrics)
    logger.info(f"Drift detected: {drift_detected}")

if __name__ == "__main__":
    drift_monitoring_flow()
