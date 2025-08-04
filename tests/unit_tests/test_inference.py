import pandas as pd
import pytest
from unittest import mock
from pipeline.inference import load_features, load_model, predict_and_save
import xgboost as xgb
import pickle


@pytest.fixture
def sample_features_df():
    data = {
        "datetime": pd.date_range("2025-08-01 00:00:00", periods=2, freq="4H"),
        "group_key": ["A001_R001", "A001_R001"],
        "entries_4h_last_week": [100, 120],
        "entries_4h_last_day": [90, 100],
        "rolling_mean_prev_day": [95, 110],
        "hour": [0, 4],
        "day_of_week": [4, 4],
    }
    df = pd.DataFrame(data)
    df["group_key"] = df["group_key"].astype("category")
    return df


def test_load_features():
    with mock.patch("pipeline.inference.create_engine") as mock_engine:
        mock_conn = mock.MagicMock()
        mock_engine.return_value = mock_conn
        mock_conn.connect.return_value = mock_conn
        mock_df = pd.DataFrame(
            {
                "datetime": ["2025-08-01 00:00:00"],
                "group_key": ["A001_R001"],
            }
        )
        mock_df["datetime"] = pd.to_datetime(mock_df["datetime"])
        mock_df["group_key"] = mock_df["group_key"].astype("category")

        with mock.patch("pipeline.inference.pd.read_sql", return_value=mock_df):
            df = load_features("user", "pass", "host", "5432", "db", "table")
            assert not df.empty
            assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
            assert pd.api.types.is_categorical_dtype(df["group_key"])


def test_load_model(tmp_path):
    # Фиктивные данные
    X = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [0, 1, 0, 1],
        }
    )
    y = [10, 20, 30, 40]

    model = xgb.XGBRegressor()
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    loaded_model = load_model(model_output_dir=str(model_path))

    assert isinstance(loaded_model, xgb.XGBRegressor)

    model_path.unlink()
    assert not model_path.exists()


def test_predict_and_save(sample_features_df):
    model = mock.MagicMock()
    model.predict.return_value = [123.0, 456.0]

    with mock.patch("pipeline.inference.create_engine") as mock_engine:
        mock_conn = mock.MagicMock()
        mock_engine.return_value = mock_conn
        mock_conn.connect.return_value = mock_conn

        with mock.patch("pipeline.inference.xgb.DMatrix"):
            predict_and_save(
                sample_features_df,
                model,
                "user",
                "pass",
                "host",
                "5432",
                "db",
                "output_table",
            )
            assert model.predict.called
