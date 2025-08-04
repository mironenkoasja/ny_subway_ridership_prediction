import pandas as pd
from pipeline.preprocess import preprocess, add_features


def sample_raw_data():
    return pd.DataFrame(
        [
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "00:00:00",
                "entries": "100",
                "exits": "50",
            },
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "04:00:00",
                "entries": "200",
                "exits": "100",
            },
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "08:00:00",
                "entries": "250",
                "exits": "150",
            },
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "12:00:00",
                "entries": "300",
                "exits": "200",
            },
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "16:00:00",
                "entries": "500",
                "exits": "300",
            },
            {
                "c_a": "A001",
                "unit": "R001",
                "scp": "01-00-00",
                "date": "2025-08-01",
                "time": "20:00:00",
                "entries": "700",
                "exits": "500",
            },
        ]
    )


def test_preprocess():
    df_raw = sample_raw_data()
    df_agg = preprocess(df_raw)

    assert not df_agg.empty
    assert set(df_agg.columns) == {"group_key", "datetime", "ridership_4h"}
    assert df_agg["ridership_4h"].sum() > 0
    assert df_agg["group_key"].str.startswith("A001_R001").all()


def test_add_features_train():
    df_raw = sample_raw_data()
    df_agg = preprocess(df_raw)
    df_train = add_features(df_agg, mode="train")

    assert "entries_4h_last_week" in df_train.columns
    assert "entries_4h_last_day" in df_train.columns
    assert "rolling_mean_prev_day" in df_train.columns
    assert df_train["hour"].between(0, 23).all()
    assert df_train["day_of_week"].between(0, 6).all()


def test_add_features_inference():
    df_raw = sample_raw_data()
    df_agg = preprocess(df_raw)
    df_inf = add_features(df_agg, mode="inference")

    assert not df_inf.empty
    assert (
        df_inf["datetime"].dt.date == df_inf["datetime"].dt.normalize().max().date()
    ).all()
    assert "entries_4h_last_week" in df_inf.columns
    assert "entries_4h_last_day" in df_inf.columns
