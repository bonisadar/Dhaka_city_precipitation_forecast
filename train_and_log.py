import os
assert os.path.exists(__file__), f"{__file__} not found!"

# Set dynamically from env or fallback
os.environ["PREFECT_API_URL"] = os.getenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")

from prefect_gcp import GcpCredentials
gcp_credentials_block = GcpCredentials.load("gcp-credentials")

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from datetime import datetime
from prefect import flow, task
from google.cloud import storage
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Download data from GCS
@task
def download_from_gcs(blob_name, local_path):
    bucket_name = "mlops-zoomcamp-bucke-2"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")
    return local_path

# (Optional) Store y_pred.txt outside MLflow too (in GCS)
# If want to avoid pulling from MLflow (e.g., for faster access or visualization), just upload to GCS directly.
@task
def upload_y_pred_to_gcs(local_file, destination_blob_name):
    bucket_name = "mlops-zoomcamp-bucke-2"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to gs://{bucket_name}/{destination_blob_name}")

@task
def fetch_last_month_y_pred(experiment_name="dhaka_city_precipitation_forecast_v7"):
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("No experiment found.")
        return None
    
    # Get runs sorted by start time descending
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"],
                              max_results=2)

    if len(runs) < 2:
        print("Not enough historical runs for drift detection.")
        return None
    
    # Take the *previous* run (second latest)
    last_run_id = runs[1].info.run_id
    
    local_y_pred_path = f"/tmp/y_pred_last_month.txt"
    client.download_artifacts(last_run_id, "y_pred.txt", "/tmp")

    return local_y_pred_path

# STEP 2: Feature engineering
@task
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
    df = df.drop(columns=['time'])

    X = df.drop(columns=['precipitation'])
    y = df['precipitation']
    return X, y

# STEP 3: Train and log with MLflow
@task
def train_and_log_model(X, y):
    # Tracking URI - use environment variable for flexibility
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("dhaka_city_precipitation_forecast_v7")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [7, 10],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3)

    with mlflow.start_run():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        signature = infer_signature(X_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({
            "mae": mae,
            "mse": mse,
            "r2": r2
        })
        mlflow.log_params(search.best_params_)

        with open("features.txt", "w") as f:
            f.write("\n".join(X.columns))
        mlflow.log_artifact("features.txt")

        np.savetxt("y_pred.txt", y_pred)
        mlflow.log_artifact("y_pred.txt")
        
        # avoid pulling from MLflow (e.g., for faster access or visualization), just upload to  GCS directly.
        
        month_str = datetime.now().strftime("%B%Y")
        upload_y_pred_to_gcs("y_pred.txt", f"predictions/y_pred_{month_str}.txt")

        # upload_y_pred_to_gcs("y_pred.txt", "predictions/y_pred_July2025.txt") 

        # Then later you can fetch it via
        # blob.download_to_filename("y_pred_last_month.txt")

        mlflow.xgboost.log_model(best_model, name="model",
                                 input_example=X_test.iloc[:5],
                                 signature=signature)

    print(f"Model trained and logged: MAE={mae:.4f}, R²={r2:.4f}")
    return {"mae": mae, "mse": mse, "r2": r2}


@task
def assign_champion_alias(model_name: str):
    client = MlflowClient()
    latest = client.get_latest_versions(model_name, stages=[])
    version = sorted(latest, key=lambda mv: int(mv.version))[-1]
    client.set_registered_model_alias(model_name, "champion", version.version)


@task
def push_metrics_to_prometheus(metrics):
    registry = CollectorRegistry()
    g_mae = Gauge('model_mae', 'Model MAE', registry=registry)
    g_r2 = Gauge('model_r2', 'Model R²', registry=registry)

    g_mae.set(metrics['mae'])
    g_r2.set(metrics['r2'])

    push_to_gateway('http://localhost:9091', job='dhaka_weather_model', registry=registry)

@task
def detect_drift(current_preds, last_month_preds, threshold=0.3):
    curr_mean = np.mean(current_preds)
    last_mean = np.mean(last_month_preds)
    drift = abs(curr_mean - last_mean) / (abs(last_mean) + 1e-6)
    
    if drift > threshold:
        print(f"Drift detected! Current mean: {curr_mean:.2f}, Last: {last_mean:.2f}")
        return True
    else:
        print("No significant drift.")
        return False

@flow(name="train_and_log_flow")
def train_and_log_flow():
    blob_name = 'raw/raw_dhaka_weather.csv'
    local_path = '/tmp/latest_weather.csv'

    file_path = download_from_gcs(blob_name, local_path)
    df = pd.read_csv(file_path)
    X, y = engineer_features(df)
    # train_and_log_model(X, y) 

    metrics = train_and_log_model(X, y)
    
    # Load last month prediction
    last_y_pred_path = fetch_last_month_y_pred()
    if last_y_pred_path and os.path.exists(last_y_pred_path):
        last_y_pred = np.loadtxt(last_y_pred_path)
        # detect_drift(metrics["mae"], last_y_pred)

        y_pred = np.loadtxt("y_pred.txt")  # Load current predictions
        detect_drift(y_pred, last_y_pred)


if __name__ == "__main__":
    train_and_log_flow()
