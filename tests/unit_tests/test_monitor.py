import pandas as pd
from unittest import mock
from pipeline import monitor


@mock.patch("pipeline.monitor.requests.get")
def test_fetch_actual_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [
        {
            "date": "2025-08-01",
            "time": "04:00:00",
            "entries": "1000",
            "exits": "900",
            "c_a": "A001",
            "unit": "R001",
            "scp": "01-00-00",
        },
        {
            "date": "2025-08-01",
            "time": "08:00:00",
            "entries": "1500",
            "exits": "1300",
            "c_a": "A001",
            "unit": "R001",
            "scp": "01-00-00",
        },
    ]
    df = monitor.fetch_actual_data("2025-08-01")
    assert isinstance(df, pd.DataFrame)
    assert "actual_ridership" in df.columns
    assert df["actual_ridership"].iloc[0] > 0


@mock.patch("pipeline.monitor.create_engine")
def test_load_predictions(mock_engine):
    mock_conn = mock.Mock()
    mock_engine.return_value = mock_conn
    mock_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-08-01", periods=2, freq="H"),
            "group_key": ["A001_R001", "A002_R002"],
            "predicted_ridership": [100, 200],
        }
    )
    pd.read_sql = mock.Mock(return_value=mock_df)

    df = monitor.load_predictions(
        "user", "pass", "host", "5432", "db", "preds", "2025-08-01"
    )
    assert isinstance(df, pd.DataFrame)
    assert "predicted_ridership" in df.columns


@mock.patch("pipeline.monitor.create_engine")
@mock.patch("pandas.DataFrame.to_sql")
def test_save_monitoring_data(mock_to_sql, mock_create_engine):
    mock_engine = mock.Mock()
    mock_create_engine.return_value = mock_engine

    df_local = pd.DataFrame(
        {
            "group_key": ["A001_R001"],
            "monitoring_date": [pd.to_datetime("2025-08-01").date()],
            "n_obs": [10],
            "mae": [5.0],
            "rmse": [6.0],
            "ridership_mean": [100],
            "load_class": ["mid-low"],
        }
    )

    df_global = pd.DataFrame(
        {
            "monitoring_date": [pd.to_datetime("2025-08-01").date()],
            "n_obs": [10],
            "mae": [5.0],
            "rmse": [6.0],
            "mean_ridership": [100],
        }
    )

    monitor.save_monitoring_data(
        df_local, df_global, "u", "p", "h", "5432", "db", "monitor_table"
    )

    # Проверяем, что to_sql был вызван дважды: для локальных и глобальных метрик
    assert mock_to_sql.call_count == 2
    mock_to_sql.assert_any_call(
        name="monitor_table", con=mock_engine, if_exists="append", index=False
    )
    mock_to_sql.assert_any_call(
        name="monitor_table_global", con=mock_engine, if_exists="append", index=False
    )
