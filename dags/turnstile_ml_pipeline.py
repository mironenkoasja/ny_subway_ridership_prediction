from airflow import DAG
from airflow.operators.bash import BashOperator
import os
import pendulum


def require_env(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value


PG_USER = require_env("PG_USER")
PG_PASSWORD = require_env("PG_PASSWORD")
PG_HOST = require_env("PG_HOST")
PG_PORT = require_env("PG_PORT")
PG_DB_NAME = require_env("PG_DB_NAME")


now = pendulum.now("UTC")
reference_date = now.replace(year=2022)


with DAG(
    dag_id="turnstile_ml_pipeline",
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"),
    catchup=False,
) as dag:

    download_and_ingest = BashOperator(
        task_id="download_and_ingest",
        bash_command="""
            python /opt/airflow/pipeline/download_ingest.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --table_name raw_turnstile_data
        """,
    )

    preprocess_train = BashOperator(
        task_id="preprocess_train",
        bash_command="""
            python /opt/airflow/pipeline/preprocess.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --raw_table raw_turnstile_data \
                --features_table turnstile_features \
                --mode train
        """,
    )

    tune_model = BashOperator(
        task_id="tune_model",
        bash_command="""
            python /opt/airflow/pipeline/tune_model.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --features_table turnstile_features \
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
            python /opt/airflow/pipeline/train_register.py \
                            --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --features_table turnstile_features \
        """,
    )

    preprocess_inference = BashOperator(
        task_id="preprocess_inference",
        bash_command="""
            python /opt/airflow/pipeline/preprocess.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --raw_table raw_turnstile_data \
                --features_table turnstile_features \
                --inference_table inference_features \
                --mode inference
        """,
    )

    predict = BashOperator(
        task_id="run_inference",
        bash_command="""
            python /opt/airflow/pipeline/inference.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --inference_table inference_features \
                --predictions_table predictions
        """,
    )

    monitoring = BashOperator(
        task_id="run_monitoring",
        bash_command="""
            python /opt/airflow/pipeline/monitor.py \
                --user """
        + PG_USER
        + """ \
                --password """
        + PG_PASSWORD
        + """ \
                --host """
        + PG_HOST
        + """ \
                --port """
        + PG_PORT
        + """ \
                --db_name """
        + PG_DB_NAME
        + """ \
                --predictions_table predictions \
                --monitoring_table monitoring \
                --reference_date """
        + reference_date.to_date_string()
        + """
        """,
    )

    download_and_ingest >> preprocess_train >> tune_model >> train_model
    train_model >> preprocess_inference >> predict >> monitoring
