# app/main.py
from fastapi import FastAPI
from app.predict import forecast_next_24_hours

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🌦️ Welcome to the Dhaka City Precipitation Forecast API!"}

@app.get("/predict")
def predict():
    return forecast_next_24_hours()
