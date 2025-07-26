# app/predict.py
from app.model import load_model
from app.utils import fetch_weather, engineer_features

model = load_model()

def forecast_next_24_hours():
    df = fetch_weather()
    X = engineer_features(df)
    preds = model.predict(X)
    df['predicted_precipitation'] = preds
    return df[['predicted_precipitation']].round(2).to_dict(orient="records")
