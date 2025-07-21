from prefect import flow
import subprocess

@flow(name="drift-monitoring-flow")
def drift_monitoring_flow():
    print("🚀 Starting drift monitoring...")
    result = subprocess.run(["python3", "monitor_drift.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ Errors:")
        print(result.stderr)


if __name__ == "__main__":
    drift_monitoring_flow()
