from prefect import flow, task
from prefect.blocks.system import Secret
from data_fetcher import fetch_weather_data, get_dynamic_date_range
from google.cloud import storage
import os

@task
def save_to_local(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path

@task
def upload_to_gcs(file_path, destination_blob_name, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)  
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}")


@flow
def fetch_and_upload_flow():
    bucket_name = "mlops-zoomcamp-bucke-2"
    # Weather parameters
    hourly_vars = [
        'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
        'apparent_temperature', 'cloudcover', 'cloudcover_low',
        'windspeed_10m', 'winddirection_10m', 'surface_pressure',
        'vapour_pressure_deficit', 'weathercode', 'wet_bulb_temperature_2m',
        'precipitation', 'is_day'
    ]

    # Dynamic date range (last 20 years)
    start_date, end_date = get_dynamic_date_range(days_back=7300, buffer_days=2)

    # Fetch data
    df = fetch_weather_data(23.8041, 90.4152, hourly_vars, start_date, end_date)

    # Local save
    local_file = f"../data/raw_dhaka_weather_{start_date}_to_{end_date}.csv"
    save_to_local(df, local_file)

    # Load GCS bucket name from Prefect Secret
    # bucket_name = Secret.load("gcp-bucket-name").get()

    # Upload to GCS
    upload_to_gcs(local_file, f'raw/raw_dhaka_weather.csv', bucket_name)

if __name__ == "__main__":
    fetch_and_upload_flow()

