from prefect import flow, task
from google.cloud import storage
import pandas as pd
import numpy as np
import os
import mlflow
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
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model and search space
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [400],
        'max_depth': [7, 10],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3)
    
    # --- MLflow Setup ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("dhaka_city_precipitation_forecast_v2")

    # Optionally: Enable autologging for XGBoost
    # mlflow.xgboost.autolog()

    with mlflow.start_run():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Infer model signature
        signature = infer_signature(X_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics and params (autolog does most, but you can add anything extra)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_params(search.best_params_)

        # Log feature names as artifact for reproducibility
        feature_path = "features.txt"
        with open(feature_path, "w") as f:
            f.write("\n".join(X.columns))
        mlflow.log_artifact(feature_path)

        # Log predictions
        np.savetxt("y_pred.txt", y_pred)
        mlflow.log_artifact("y_pred.txt")

        # (Optional) Log the model explicitly if not using autolog
        mlflow.xgboost.log_model(best_model, name="model",
                                 input_example=X_test.iloc[:5],
                                 signature=signature)
        mlflow.log_metric("mae", mae)
        mlflow.log_params(search.best_params_)              

    print(f"Model trained and logged: MAE={mae:.4f}, R²={r2:.4f}")
    return {"mae": mae, "mse": mse, "r2": r2}

# Main Prefect flow
@flow
def train_and_log_flow():
    blob_name = 'raw/raw_dhaka_weather.csv'
    local_path = '/tmp/latest_weather.csv'

    # Download data
    file_path = download_from_gcs(blob_name, local_path)

    # Feature engineering
    df = pd.read_csv(file_path)
    X, y = engineer_features(df)

    # Train + log
    train_and_log_model(X, y)

if __name__ == "__main__":
    train_and_log_flow()
