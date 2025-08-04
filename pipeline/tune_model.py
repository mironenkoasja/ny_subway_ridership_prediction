import os
import argparse
import pandas as pd
import numpy as np
import mlflow
from sqlalchemy import create_engine
from datetime import datetime
import json
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


def load_features(user, password, host, port, db_name, table_name):
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["group_key"] = df["group_key"].astype("category")
    return df


def tune(df, experiment_name, run_prefix, n_trials, tracking_uri=None):
    CUTOFF_DATE = df["datetime"].max() - pd.Timedelta(days=1)
    train = df[df["datetime"] < CUTOFF_DATE]
    val = df[df["datetime"] >= CUTOFF_DATE]

    print("Train shape:", train.shape)
    print("Validation shape:", val.shape)
    print("Max datetime in dataset:", df["datetime"].max())

    FEATURES = [
        "entries_4h_last_week",
        "entries_4h_last_day",
        "rolling_mean_prev_day",
        "hour",
        "day_of_week",
        "group_key",
    ]
    TARGET = "ridership_4h"

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val = val[FEATURES]
    y_val = val[TARGET]

    for col in [
        "entries_4h_last_week",
        "entries_4h_last_day",
        "rolling_mean_prev_day",
        "hour",
        "day_of_week",
    ]:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
    X_train["group_key"] = X_train["group_key"].astype("category")
    X_val["group_key"] = X_val["group_key"].astype("category")

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -3, 1),
        "reg_lambda": hp.loguniform("reg_lambda", -3, 1),
        "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
    }

    def objective(params):
        with mlflow.start_run(
            run_name=f"{run_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)

            model = xgb.train(params, dtrain)

            y_pred = model.predict(dval)
            rmse = root_mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            return {"loss": rmse, "status": STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=Trials(),
        rstate=np.random.default_rng(42),
    )

    print("Best hyperparameters:", best_result)

    with mlflow.start_run(
        run_name=f"log-best-params-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        nested=True,
    ):

        tmp_dir = "/opt/airflow/shared"
        os.makedirs(tmp_dir, exist_ok=True)
        params_path = os.path.join(tmp_dir, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_result, f)
        # mlflow.log_artifact(params_path, artifact_path="best_params")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--features_table", required=True)
    parser.add_argument("--experiment_name", default="xgboost-hyperopt")
    parser.add_argument("--run_prefix", default="tune")
    parser.add_argument("--n_trials", type=int, default=30)

    args = parser.parse_args()

    df = load_features(
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        db_name=args.db_name,
        table_name=args.features_table,
    )

    tune(df, args.experiment_name, args.run_prefix, args.n_trials, tracking_uri="http://mlflow:5000")
