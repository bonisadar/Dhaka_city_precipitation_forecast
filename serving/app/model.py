# app/model.py
import mlflow
from mlflow.tracking import MlflowClient

def get_champion_metrics(model_name="dhaka_city_precipitation_xgb"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # <-- Make sure URI is set
    client = MlflowClient()

    # Use alias resolution (champion) for clarity
    model_version = client.get_model_version_by_alias(model_name, "champion")
    run_id = model_version.run_id
    run = client.get_run(run_id)
    return run.data.metrics

