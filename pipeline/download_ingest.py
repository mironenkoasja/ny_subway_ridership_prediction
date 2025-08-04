import pandas as pd
import requests
from sqlalchemy import create_engine
import argparse
import pendulum


def fetch_turnstile_data(date: str) -> pd.DataFrame:
    url = f"https://data.ny.gov/resource/k7j9-jnct.json?date={date}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df
    print(f"[WARN] Failed to load or empty data for {date}")
    return pd.DataFrame()


def generate_dates(start_date: str, end_date: str) -> list[str]:
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    return [d.strftime("%Y-%m-%d") for d in date_range]


def download_and_ingest(
    start_date, end_date, user, password, host, port, db_name, table_name
):
    dates = generate_dates(start_date, end_date)

    print(f"[INFO] Fetching data for dates: {dates}")
    dfs = [fetch_turnstile_data(date) for date in dates]
    df_raw = pd.concat(dfs, ignore_index=True)

    if df_raw.empty:
        print("[ERROR] No data fetched. Exiting.")
        return

    df_raw["entries"] = pd.to_numeric(df_raw["entries"], errors="coerce")
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    df_raw.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    print(f"[OK] Ingested {len(df_raw)} rows into table '{table_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--table_name", required=True)
    args = parser.parse_args()

    # Current date turn into 2022 year
    now = pendulum.now("UTC")
    print(now)
    reference_date = now.replace(year=2022)

    end_date = reference_date.subtract(days=1)
    start_date = reference_date.subtract(days=14)

    download_and_ingest(start_date=start_date, end_date=end_date, **vars(args))
