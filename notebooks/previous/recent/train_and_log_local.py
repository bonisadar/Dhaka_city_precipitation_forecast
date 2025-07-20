import mlflow.xgboost
from mlflow.models import infer_signature

# After model is trained and predictions made
y_pred = best_model.predict(X_test)

# Infer model signature
signature = infer_signature(X_test, y_pred)

# Log model manually with signature and input example
mlflow.xgboost.log_model(
    best_model,
    artifact_path="model",
    input_example=X_test.iloc[:5],
    signature=signature
)
