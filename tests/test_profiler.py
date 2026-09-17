import pandas as pd
import pytest

from modules.profiler import profile_dataset


def test_basic_dataset_profile():
    df = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "income": [50000, 70000, 90000],
            "city": ["Fullerton", "Anaheim", "Irvine"],
            "target": [0, 1, 0],
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["rows"] == 3
    assert profile["columns"] == 4
    assert profile["column_names"] == ["age", "income", "city", "target"]
    assert profile["duplicate_rows"] == 0


def test_numerical_and_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "city": ["Fullerton", "Anaheim"],
            "target": [0, 1],
        }
    )

    profile = profile_dataset(df, "target")

    assert "age" in profile["numerical_columns"]
    assert "target" in profile["numerical_columns"]
    assert "city" in profile["categorical_columns"]


def test_missing_values():
    df = pd.DataFrame(
        {
            "age": [25, None, 45],
            "target": [0, 1, None],
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["missing_values"]["age"] == 1
    assert profile["missing_values"]["target"] == 1
    assert profile["target_missing_values"] == 1


def test_duplicate_rows():
    df = pd.DataFrame(
        {
            "age": [25, 25, 35],
            "target": [0, 0, 1],
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["duplicate_rows"] == 1


def test_target_information():
    df = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["target_column"] == "target"
    assert profile["target_unique_values"] == 2


def test_missing_target_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
        }
    )

    with pytest.raises(ValueError, match="Target column"):
        profile_dataset(df, "target")


def test_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataset is empty"):
        profile_dataset(df, "target")


def test_invalid_input():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        profile_dataset(["not", "a", "dataframe"], "target")