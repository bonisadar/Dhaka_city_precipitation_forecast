# app/utils.py
import pandas as pd
import numpy as np
import requests

def fetch_weather():
    params = {
        "latitude": 23.8103,
        "longitude": 90.4125,
        "hourly": ",".join([
            'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
            'apparent_temperature', 'cloudcover', 'cloudcover_low',
            'windspeed_10m', 'winddirection_10m', 'surface_pressure',
            'vapour_pressure_deficit', 'weathercode', 'wet_bulb_temperature_2m', 'is_day'
        ]),
        "forecast_days": 1,
        "timezone": "Asia/Dhaka"
    }

    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])

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

    for col in [col for col in df.columns if 'temp_lag_' in col]:
        df[col] = df[col].fillna(df['temperature_2m'].iloc[0])

    return df
