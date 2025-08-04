import os
import contextlib
from unittest import mock
import pandas as pd

from pipeline import train_register


@mock.patch("sqlalchemy.create_engine")
def test_load_features(mock_engine):
    df_mock = pd.DataFrame(
        {
            "datetime": ["2025-08-01", "2025-08-02"],
            "group_key": ["A001_R001", "A002_R002"],
        }
    )
    mock_conn = mock.MagicMock()
    mock_engine.return_value.connect.return_value = mock_conn
    mock_engine.return_value.__enter__.return_value = mock_conn
    mock_engine.return_value.__exit__.return_value = False

    with mock.patch("pandas.read_sql", return_value=df_mock):
        df = train_register.load_features(
            "user", "pass", "localhost", "5432", "testdb", "features"
        )

    assert isinstance(df, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert pd.api.types.is_categorical_dtype(df["group_key"])


@mock.patch("pipeline.train_register.MlflowClient.get_experiment_by_name")
@mock.patch("pipeline.train_register.mlflow.search_runs")
@mock.patch("pipeline.train_register.mlflow.set_experiment")
@mock.patch("pipeline.train_register.mlflow.log_params")
@mock.patch("pipeline.train_register.mlflow.set_tag")
@mock.patch("pipeline.train_register.get_best_params")
def test_train(
    mock_get_best_params,
    mock_set_tag,
    mock_log_params,
    mock_set_experiment,
    mock_search_runs,
    mock_get_experiment_by_name,
    tmp_path,
    monkeypatch,
):
    model_output_path = tmp_path / "model.pkl"

    mock_get_best_params.return_value = {
        "max_depth": 3,
        "min_child_weight": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "objective": "reg:squarederror",
        "seed": 42,
        "verbosity": 0,
    }

    mock_get_experiment_by_name.return_value = mock.Mock(experiment_id="123")
    fake_run = mock.Mock()
    fake_run.info.run_id = "fake-run-id"
    mock_search_runs.return_value = [fake_run]

    monkeypatch.setattr(
        train_register.mlflow,
        "start_run",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-08-01", periods=5, freq="4H"),
            "entries_4h_last_week": [1.0, 2.0, 3.0, 4.0, 5.0],
            "entries_4h_last_day": [1.0] * 5,
            "rolling_mean_prev_day": [1.0] * 5,
            "hour": [0] * 5,
            "day_of_week": [1] * 5,
            "group_key": ["A001_R001"] * 5,
            "ridership_4h": [100] * 5,
        }
    )
    df["group_key"] = df["group_key"].astype("category")

    train_register.train(df, model_output_dir=str(model_output_path))

    assert os.path.exists(model_output_path)
    os.remove(model_output_path)
