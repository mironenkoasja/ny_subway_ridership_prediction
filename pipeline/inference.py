import argparse
import pandas as pd
from sqlalchemy import create_engine
import mlflow
import mlflow.xgboost
import xgboost as xgb
import os
import pickle

def load_features(user, password, host, port, db_name, inference_table):
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    query = f"SELECT * FROM {inference_table}"
    df = pd.read_sql(query, con=engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["group_key"] = df["group_key"].astype("category")
    return df

def load_model(model_uri="models:/xgboost-best-model/Production"):
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_output_dir = "/opt/airflow/shared/xgboost_model/model.pkl"
    with open(model_output_dir, "rb") as f:
        model = pickle.load(f)
    return model

def predict_and_save(df, model, user, password, host, port, db_name, output_table):
    FEATURES = ["entries_4h_last_week", "entries_4h_last_day", "rolling_mean_prev_day", "hour", "day_of_week", "group_key"]

    df_pred = df.copy()
    df_pred["entries_4h_last_week"] = df_pred["entries_4h_last_week"].astype(float)
    dmat = xgb.DMatrix(df_pred[FEATURES], enable_categorical=True)
    df_pred["predicted_ridership"] = model.predict(dmat)

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    df_pred[["datetime", "group_key", "predicted_ridership"]].to_sql(
        output_table, con=engine, if_exists="append", index=False
    )
    print(f"[OK] Saved {len(df_pred)} predictions to '{output_table}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--inference_table", required=True)
    parser.add_argument("--predictions_table", required=True)
    parser.add_argument("--model_uri", default="models:/xgboost-best-model/Production", help="MLflow model URI")

    args = parser.parse_args()

    df_features = load_features(
        args.user, args.password, args.host, args.port, args.db_name, args.inference_table
    )
    print(df_features.head(2))
    model = load_model(args.model_uri)
    predict_and_save(
        df_features, model,
        args.user, args.password, args.host, args.port,
        args.db_name, args.predictions_table
    )
