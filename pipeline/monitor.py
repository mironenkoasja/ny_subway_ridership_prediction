import argparse
import pandas as pd
import requests
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime


def fetch_actual_data(date: str) -> pd.DataFrame:
    url = f"https://data.ny.gov/resource/k7j9-jnct.json?date={date}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data for {date}: {response.status_code}")
    df = pd.DataFrame(response.json())

    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"], errors="coerce"
    )
    df["entries"] = pd.to_numeric(df["entries"], errors="coerce")
    df["exits"] = pd.to_numeric(df["exits"], errors="coerce")

    df = df.dropna(subset=["datetime", "entries",
                   "exits", "c_a", "unit", "scp"])
    df = df.sort_values(["c_a", "unit", "scp", "datetime"])

    df["entries_diff"] = df.groupby(["c_a", "unit", "scp"])["entries"].diff()
    df["exits_diff"] = df.groupby(["c_a", "unit", "scp"])["exits"].diff()
    df["ridership"] = df["entries_diff"] + df["exits_diff"]
    df = df[(df["ridership"] >= 0) & (df["ridership"] < 5000)]
    df["group_key"] = df["c_a"] + "_" + df["unit"]

    agg = (
        df.groupby(["group_key", "datetime"])["ridership"]
        .sum()
        .reset_index()
        .rename(columns={"ridership": "actual_ridership"})
    )
    return agg

def load_predictions(
    user, password, host, port, db_name, predictions_table, target_date
):
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    query = f"""
        SELECT datetime, group_key, predicted_ridership 
        FROM {predictions_table} 
        WHERE DATE(datetime) = '{target_date}'
    """
    df = pd.read_sql(query, con=engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def save_monitoring_data(
    df, df_global, user, password, host, port, db_name, monitoring_table
):
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    df.to_sql(name=monitoring_table, con=engine,
              if_exists="append", index=False)

    df_global.to_sql(
        name=monitoring_table + "_global",
          con=engine, 
          if_exists="append", index=False
    )

    print(f"[OK] Saved {len(df)} rows to '{monitoring_table}'")


def run_monitoring(
    user,
    password,
    host,
    port,
    db_name,
    predictions_table,
    monitoring_table,
    reference_date,
):
    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    print(f"[Monitoring] For date: {reference_date}")

    df_pred = load_predictions(
        user, password, host, port, db_name, predictions_table, reference_date
    )
    df_actual = fetch_actual_data(str(reference_date))
    print(reference_date)

    df = pd.merge(df_pred, df_actual, on=[
                  "datetime", "group_key"], how="inner")

    df["abs_error"] = (df["actual_ridership"] -
                       df["predicted_ridership"]).abs()
    df["squared_error"] = (df["actual_ridership"] -
                           df["predicted_ridership"]) ** 2

    df_metrics = (
        df.groupby("group_key")
        .agg(
            monitoring_date=("datetime", lambda x: x.min().date()),
            n_obs=("datetime", "count"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(x.mean())),
            ridership_mean=("actual_ridership", "mean"),
        )
        .reset_index()
    )
    df_metrics["load_class"] = pd.qcut(
        df_metrics["ridership_mean"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"]
    )

    df_global_metrics = pd.DataFrame(
        [
            {
                "monitoring_date": df["datetime"].min().date(),
                "n_obs": len(df),
                "mae": df["abs_error"].mean(),
                "rmse": np.sqrt(df["squared_error"].mean()),
                "mean_ridership": df["actual_ridership"].mean(),
            }
        ]
    )

    print(df_global_metrics.head(5))
    print(df_metrics.head(5))

    save_monitoring_data(
        df_metrics,
        df_global_metrics,
        user,
        password,
        host,
        port,
        db_name,
        monitoring_table,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--predictions_table", required=True)
    parser.add_argument("--monitoring_table", required=True)
    parser.add_argument("--reference_date", required=True)

    args = parser.parse_args()

    run_monitoring(
        args.user,
        args.password,
        args.host,
        args.port,
        args.db_name,
        args.predictions_table,
        args.monitoring_table,
        args.reference_date,
    )
