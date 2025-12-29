from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="credit_model_retraining",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    retrain = BashOperator(
        task_id="retrain_model",
        bash_command="echo 'Retraining model...'"
    )

    retrain
