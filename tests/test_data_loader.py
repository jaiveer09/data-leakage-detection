import pandas as pd
import pytest

from modules.data_loader import load_csv, validate_columns


def test_load_valid_csv(tmp_path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "age,income,target\n"
        "25,50000,0\n"
        "35,70000,1\n"
    )

    df = load_csv(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["age", "income", "target"]


def test_load_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("age,income,target\n")

    with pytest.raises(ValueError, match="CSV file is empty"):
        load_csv(csv_file)


def test_validate_columns_valid():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "timestamp": ["2026-01-01", "2026-01-02"],
            "customer_id": [101, 102],
            "target": [0, 1],
        }
    )

    result = validate_columns(
        df,
        target_column="target",
        timestamp_column="timestamp",
        group_column="customer_id",
    )

    assert result is True


def test_missing_target_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "income": [50000, 70000],
        }
    )

    with pytest.raises(ValueError, match="target_column"):
        validate_columns(df, target_column="target")


def test_target_column_required():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="target_column is required"):
        validate_columns(df, target_column=None)


def test_invalid_timestamp_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="timestamp_column"):
        validate_columns(
            df,
            target_column="target",
            timestamp_column="timestamp",
        )


def test_invalid_group_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="group_column"):
        validate_columns(
            df,
            target_column="target",
            group_column="customer_id",
        )


def test_optional_columns_not_required():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    result = validate_columns(
        df,
        target_column="target",
    )

    assert result is True