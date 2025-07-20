Project Description
This project aims to build and train a XGBoost Regression model to predict Dhaka city precipitation. 
The training data is collect through open-meteo api. The model is train on 20 years of historical data. GCP is used to store the training data. A GCP hosted Postgresql server is used as a backend storage for Mlflow artifacts and runs. A local Prefect server is used for orchestration.

To reproduce the steps you must have a GCS bucket, A Postgresql 17/16 instance in GCP with a database, service account in GCP with proper role granted (storage admin), service account key.

When connecting to datasource in grafana pick prometheus and then server URL http://your local machine ip:9090
