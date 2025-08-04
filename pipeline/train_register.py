import json
import mlflow
import argparse
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sqlalchemy import create_engine
from mlflow.tracking import MlflowClient

TABLE_NAME = "turnstile_features"
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_HYPEROPT = "xgboost-hyperopt"
EXPERIMENT_TRAIN = "xgboost-best-model"


def load_features(user, password, host, port, db_name, table_name):
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["group_key"] = df["group_key"].astype("category")
    return df


def get_best_params():
    params_path = "/opt/airflow/shared/best_params.json"
    with open(params_path, "r") as f:
        best_params_raw = json.load(f)

    return {
        "max_depth": int(best_params_raw["max_depth"]),
        "min_child_weight": int(best_params_raw["min_child_weight"]),
        "reg_alpha": float(best_params_raw["reg_alpha"]),
        "reg_lambda": float(best_params_raw["reg_lambda"]),
        "learning_rate": float(best_params_raw["learning_rate"]),
        "n_estimators": int(best_params_raw["n_estimators"]),
        "objective": "reg:squarederror",
        "seed": 42,
        "verbosity": 0,
    }


def train(df, model_output_dir, tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # client = MlflowClient()
    # exp = client.get_experiment_by_name(EXPERIMENT_HYPEROPT)
    # runs = client.search_runs(
    #     exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=1
    # )
    # best_run_id = runs[0].info.run_id
    FEATURES = [
        "entries_4h_last_week",
        "entries_4h_last_day",
        "rolling_mean_prev_day",
        "hour",
        "day_of_week",
        "group_key",
    ]
    TARGET = "ridership_4h"
        
    train = df.copy()
    best_params = get_best_params()

    for col in FEATURES:
        if col != "group_key":
            train[col] = pd.to_numeric(train[col], errors="coerce").replace(
                -1.0, np.nan
            )
        else:
            train[col] = train[col].astype("category")

    X_train = train[FEATURES].copy()
    y_train = train[TARGET].copy()

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    # --- Обучение и логгирование модели ---
    mlflow.set_experiment(EXPERIMENT_TRAIN)
    with mlflow.start_run():
        mlflow.set_tag("stage", "train")
        mlflow.log_params(best_params)

        model = xgb.train(best_params, dtrain)

    with open(model_output_dir, "wb") as f:
        pickle.dump(model, f)
        # mlflow.xgboost.save_model(model, path=model_output_dir)
        # mlflow.log_artifacts(model_output_dir, artifact_path="model")


# --- Точка входа ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--features_table", default=TABLE_NAME)
    args = parser.parse_args()

    df = load_features(
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        db_name=args.db_name,
        table_name=args.features_table,
    )
    model_output_dir = "/opt/airflow/shared/xgboost_model/model.pkl"
    train(df, model_output_dir, tracking_uri=MLFLOW_URI)
