import pandas as pd
import pytest

from modules.config import create_config
from modules.split_validator import validate_split_configuration


def test_valid_random_split():
    df = pd.DataFrame({
        "feature": range(10),
        "target": [0, 1] * 5,
    })

    config = create_config(
        target_column="target",
        split_type="random",
        test_size=0.2,
    )

    result = validate_split_configuration(df, config)

    assert result["valid"] is True
    assert result["split_type"] == "random"
    assert result["train_rows"] == 8
    assert result["test_rows"] == 2


def test_random_split_dataset_too_small():
    df = pd.DataFrame({
        "feature": [10, 20],
        "target": [0, 1],
    })

    config = create_config(
        target_column="target",
        split_type="random",
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="Dataset is too small",
    ):
        validate_split_configuration(df, config)


def test_valid_time_split():
    df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
        ],
        "target": [0, 1, 0, 1, 0],
    })

    config = create_config(
        target_column="target",
        timestamp_column="date",
        split_type="time",
        test_size=0.2,
    )

    result = validate_split_configuration(df, config)

    assert result["valid"] is True
    assert result["split_type"] == "time"
    assert result["train_rows"] == 4
    assert result["test_rows"] == 1
    assert result["earliest_timestamp"] == pd.Timestamp("2026-01-01")
    assert result["latest_timestamp"] == pd.Timestamp("2026-01-05")


def test_time_split_invalid_timestamp():
    df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "not-a-date",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
        ],
        "target": [0, 1, 0, 1, 0],
    })

    config = create_config(
        target_column="target",
        timestamp_column="date",
        split_type="time",
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="missing or invalid timestamp",
    ):
        validate_split_configuration(df, config)


def test_valid_group_split():
    df = pd.DataFrame({
        "customer_id": [101, 101, 102, 102, 103],
        "target": [0, 1, 0, 1, 0],
    })

    config = create_config(
        target_column="target",
        group_column="customer_id",
        split_type="group",
    )

    result = validate_split_configuration(df, config)

    assert result["valid"] is True
    assert result["split_type"] == "group"
    assert result["group_column"] == "customer_id"
    assert result["unique_groups"] == 3


def test_group_split_missing_values():
    df = pd.DataFrame({
        "customer_id": [101, 102, None, 103],
        "target": [0, 1, 0, 1],
    })

    config = create_config(
        target_column="target",
        group_column="customer_id",
        split_type="group",
    )

    with pytest.raises(
        ValueError,
        match="Group column contains missing values",
    ):
        validate_split_configuration(df, config)


def test_group_split_requires_multiple_groups():
    df = pd.DataFrame({
        "customer_id": [101, 101, 101, 101],
        "target": [0, 1, 0, 1],
    })

    config = create_config(
        target_column="target",
        group_column="customer_id",
        split_type="group",
    )

    with pytest.raises(
        ValueError,
        match="at least two unique groups",
    ):
        validate_split_configuration(df, config)