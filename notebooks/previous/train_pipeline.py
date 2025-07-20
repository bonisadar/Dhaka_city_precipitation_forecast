import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prefect import flow, task
import mlflow

# --------- Prefect Tasks ---------

@task(retries=3, retry_delay_seconds=30)
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

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
    df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
    df = df.drop(columns=['time'])
    X = df.drop(columns=['precipitation'])
    y = df['precipitation']
    return X, y

@task
def train_model(X, y, model_output_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=2)
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_tracking_uri("http://34.131.177.53:5000")  # Replace with your MLflow tracking server
    mlflow.set_experiment("dhaka-weather-forecast")

    with mlflow.start_run():
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({"mae": mae, "mse": mse, "r2": r2})
        mlflow.sklearn.log_model(search.best_estimator_, "model")

        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(search.best_estimator_, f)

    return model_output_path

# --------- Prefect Flow ---------

@flow(name="train-dhaka-weather-model")
def training_pipeline():
    df = load_data('../data/raw_dhaka_weather_2005-07-07_to_2025-07-02.csv')
    X, y = engineer_features(df)
    model_path = train_model(X, y, '../models/best_model_prefect.pkl')
    print(f"Model saved to {model_path}")
