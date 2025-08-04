import pandas as pd
from unittest.mock import patch, MagicMock
from pipeline.download_ingest import (
    generate_dates,
    fetch_turnstile_data,
    download_and_ingest,
)

# ---------- generate_dates ----------


def test_generate_dates():
    dates = generate_dates("2025-08-01", "2025-08-03")
    assert dates == ["2025-08-01", "2025-08-02", "2025-08-03"]


# ---------- fetch_turnstile_data (mocked) ----------


@patch("pipeline.download_ingest.requests.get")
def test_fetch_turnstile_data_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [
        {
            "c_a": "A002",
            "unit": "R051",
            "scp": "02-00-00",
            "line_name": "NQR456W",
            "division": "BMT",
            "date": "2022-06-22T00:00:00.000",
            "time": "00:00:00",
            "description": "REGULAR",
            "entries": "7728294",
            "exits": "2717547",
        }
    ]

    df = fetch_turnstile_data("2025-08-01")
    assert not df.empty
    assert "unit" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@patch("pipeline.download_ingest.requests.get")
def test_fetch_turnstile_data_empty(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = []

    df = fetch_turnstile_data("2025-08-01")
    assert df.empty


# ---------- download_and_ingest (mocked) ----------


@patch("pipeline.download_ingest.fetch_turnstile_data")
@patch("pipeline.download_ingest.create_engine")
def test_download_and_ingest(mock_engine, mock_fetch):
    mock_fetch.return_value = pd.DataFrame(
        {
            "c_a": ["A002"],
            "unit": ["R051"],
            "scp": ["02-00-00"],
            "line_name": ["NQR456W"],
            "division": ["BMT"],
            "date": ["2022-06-22T00:00:00.000"],
            "time": ["00:00:00"],
            "description": ["REGULAR"],
            "entries": ["7728294"],
            "exits": ["2717547"],
        }
    )

    mock_engine.return_value = MagicMock()

    download_and_ingest(
        start_date="2025-08-01",
        end_date="2025-08-01",
        user="user",
        password="pass",
        host="localhost",
        port="5432",
        db_name="testdb",
        table_name="test_table",
    )

    mock_fetch.assert_called_once_with("2025-08-01")

    mock_engine.assert_called_once_with("postgresql://user:pass@localhost:5432/testdb")
