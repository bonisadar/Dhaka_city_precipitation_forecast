from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from train_and_log import train_and_log_flow

deployment = Deployment.build_from_flow(
    flow=train_and_log_flow,
    name="monthly-model-trainer",
    schedule=(CronSchedule(cron="0 0 1 * *", timezone="Asia/Dhaka")),  # Every 1st of month at 00:00
    work_queue_name="default"
)

if __name__ == "__main__":
    deployment.apply()

