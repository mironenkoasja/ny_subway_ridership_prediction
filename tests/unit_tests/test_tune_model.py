import pandas as pd
from unittest import mock
import contextlib

from pipeline.tune_model import load_features, tune


@mock.patch("pipeline.tune_model.create_engine")
def test_load_features(mock_engine):
    df_mock = pd.DataFrame(
        {
            "datetime": ["2025-08-01", "2025-08-02"],
            "group_key": ["A001_R001", "A002_R002"],
        }
    )
    mock_engine.return_value.connect.return_value = None
    mock_engine.return_value.__enter__.return_value = None
    with mock.patch("pandas.read_sql", return_value=df_mock):
        df = load_features("user", "pass", "localhost", "5432", "testdb", "features")

    assert isinstance(df, pd.DataFrame)
    assert "datetime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert pd.api.types.is_categorical_dtype(df["group_key"])


@mock.patch("pipeline.tune_model.xgb.DMatrix")
@mock.patch("pipeline.tune_model.fmin")
@mock.patch("pipeline.tune_model.mlflow.set_experiment")
@mock.patch("pipeline.tune_model.mlflow.start_run")
def test_tune_mocked(
    mock_start_run, mock_set_experiment, mock_fmin, mock_dmatrix, monkeypatch
):
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-08-01", periods=10, freq="4H"),
            "entries_4h_last_week": [1.0] * 10,
            "entries_4h_last_day": [1.0] * 10,
            "rolling_mean_prev_day": [1.0] * 10,
            "hour": [0] * 10,
            "day_of_week": [1] * 10,
            "group_key": ["A001_R001"] * 10,
            "ridership_4h": [100] * 10,
        }
    )

    monkeypatch.setattr("pipeline.tune_model.mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("pipeline.tune_model.mlflow.log_params", lambda params: None)
    monkeypatch.setattr("pipeline.tune_model.mlflow.log_metric", lambda key, val: None)
    monkeypatch.setattr(
        "pipeline.tune_model.mlflow.start_run",
        lambda **kwargs: contextlib.nullcontext(),
    )

    mock_fmin.return_value = {"best_param": 123}

    tune(df, experiment_name="test-exp", run_prefix="test", n_trials=1)

    assert mock_fmin.called
    assert mock_set_experiment.called
