import pandas as pd
from sqlalchemy import create_engine
import argparse
from datetime import timedelta


def load_raw_data(user, password, host, port, db_name, raw_table):
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    query = f"SELECT * FROM {raw_table}"
    df = pd.read_sql(query, con=engine)
    print(f"Loaded {df.shape}")
    return df


def preprocess(df_raw):
    df_raw["entries"] = pd.to_numeric(df_raw["entries"], errors="coerce")
    df_raw["exits"] = pd.to_numeric(df_raw["exits"], errors="coerce")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["datetime"] = pd.to_datetime(
        df_raw["date"].astype(str) + " " + df_raw["time"], errors="coerce"
    )

    df_raw = df_raw.dropna(subset=["entries", "exits", "datetime", "c_a", "unit", "scp"])
    df_raw = df_raw.sort_values(["c_a", "unit", "scp", "datetime"])

    df_raw["entries_diff"] = df_raw.groupby(["c_a", "unit", "scp"])["entries"].diff()
    df_raw["exits_diff"] = df_raw.groupby(["c_a", "unit", "scp"])["exits"].diff()
    df_raw["ridership"] = df_raw["entries_diff"] + df_raw["exits_diff"]
    df_raw = df_raw[(df_raw["ridership"] >= 0) & (df_raw["ridership"] < 5000)]

    df_raw["group_key"] = df_raw["c_a"] + "_" + df_raw["unit"]

    df_agg = (
        df_raw.groupby(["group_key", "datetime"])["ridership"]
        .sum()
        .reset_index()
        .rename(columns={"ridership": "ridership_4h"})
    )
    print(f"Aggregated df {df_agg.shape}")
    return df_agg.sort_values(["group_key", "datetime"])


def add_features(df, mode="train"):
    df = df.copy()
    df["entries_4h_last_week"] = df.groupby("group_key")["ridership_4h"].shift(6 * 7)
    df["entries_4h_last_day"] = df.groupby("group_key")["ridership_4h"].shift(6)
    df["rolling_mean_prev_day"] = df.groupby("group_key")["ridership_4h"].transform(
        lambda x: x.shift(6).rolling(6).mean()
    )
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.weekday

    if mode == "train":
        return df

    elif mode == "inference":
        next_day = df["datetime"].max().normalize() + timedelta(days=1)
        hours = sorted(df["hour"].unique())
        groups = df["group_key"].unique()
        future_rows = [
            {
                "group_key": group,
                "datetime": next_day + timedelta(hours=int(hour)),
                "ridership_4h": 0,
            }
            for group in groups
            for hour in hours
        ]
        df_future = pd.DataFrame(future_rows)
        df = pd.concat([df, df_future], ignore_index=True)
        df = df.sort_values(["group_key", "datetime"])

        df["entries_4h_last_week"] = df.groupby("group_key")["ridership_4h"].shift(6 * 7)
        df["entries_4h_last_day"] = df.groupby("group_key")["ridership_4h"].shift(6)
        df["rolling_mean_prev_day"] = df.groupby("group_key")["ridership_4h"].transform(
            lambda x: x.shift(6).rolling(6).mean()
        )
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.weekday

        return df[df["datetime"].dt.normalize() == next_day]
    else:
        raise ValueError("mode must be 'train' or 'inference'")


def save_features(df, user, password, host, port, db_name, table_name):
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    print(f"[OK] Saved {len(df)} rows to '{table_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--raw_table", required=True)
    parser.add_argument("--features_table", required=True)
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument(
        "--inference_table", required=False, help="Table to save inference features"
    )

    args = parser.parse_args()

    df_raw = load_raw_data(
        args.user, args.password, args.host, args.port, args.db_name, args.raw_table
    )
    df_agg = preprocess(df_raw)
    df_features = add_features(df_agg, mode=args.mode)

    if args.mode == "inference" and args.inference_table:
        save_features(
            df_features,
            args.user,
            args.password,
            args.host,
            args.port,
            args.db_name,
            args.inference_table,
        )
    else:
        save_features(
            df_features,
            args.user,
            args.password,
            args.host,
            args.port,
            args.db_name,
            args.features_table,
        )
