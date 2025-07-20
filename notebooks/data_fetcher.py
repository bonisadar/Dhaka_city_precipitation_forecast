import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

def get_dynamic_date_range(days_back=7300, buffer_days=7):
    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=buffer_days)
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def fetch_weather_data(latitude, longitude, hourly_variables, start_date=None, end_date=None, timezone='Asia/Bangkok'):
    if start_date is None or end_date is None:
        start_date, end_date = get_dynamic_date_range(days_back=7300, buffer_days=7)

    base_url = 'https://archive-api.open-meteo.com/v1/archive'
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly_variables),
        'timezone': timezone
    }
    
    print(f"Fetching weather data from {start_date} to {end_date} for lat: {latitude}, lon: {longitude}")
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    
    return df

if __name__ == "__main__":
    hourly_vars = [
        'temperature_2m', 
        'relative_humidity_2m', 
        'dewpoint_2m',
        'apparent_temperature', 
        'cloudcover', 
        'cloudcover_low',
        'windspeed_10m', 
        'winddirection_10m', 
        'surface_pressure',
        'vapour_pressure_deficit', 
        'weathercode', 
        'wet_bulb_temperature_2m',
        'precipitation', 
        'is_day'
    ]

    df = fetch_weather_data(23.8041, 90.4152, hourly_vars)
    df['time'] = pd.to_datetime(df['time'])

    # Ensure directory exists
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'raw_dhaka_weather.csv')
    df.to_csv(output_path, index=False)

    print(f"Weather data saved: {len(df)} rows | From {df['time'].min().date()} to {df['time'].max().date()}")
