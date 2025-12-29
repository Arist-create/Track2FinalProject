from airflow import DAG
from datetime import datetime
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="credit_scoring_retraining",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
):
    start = EmptyOperator(task_id="start")
