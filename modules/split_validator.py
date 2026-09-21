import pandas as pd


def validate_split_configuration(df, config):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    split_type = config["split_type"]

    if split_type == "random":
        return validate_random_split(df, config)

    if split_type == "time":
        return validate_time_split(df, config)

    if split_type == "group":
        return validate_group_split(df, config)

    raise ValueError(f"Unsupported split type: {split_type!r}.")


def validate_random_split(df, config):
    test_size = config["test_size"]

    test_rows = int(len(df) * test_size)
    train_rows = len(df) - test_rows

    if test_rows < 1:
        raise ValueError(
            "Dataset is too small for the selected test size."
        )

    return {
        "split_type": "random",
        "valid": True,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def validate_time_split(df, config):
    timestamp_column = config["timestamp_column"]
    test_size = config["test_size"]

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} was not found."
        )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = int(timestamps.isna().sum())

    if invalid_timestamps > 0:
        raise ValueError(
            f"Timestamp column contains {invalid_timestamps} "
            "missing or invalid timestamp value(s)."
        )

    test_rows = int(len(df) * test_size)
    train_rows = len(df) - test_rows

    if test_rows < 1:
        raise ValueError(
            "Dataset is too small for the selected test size."
        )

    return {
        "split_type": "time",
        "valid": True,
        "timestamp_column": timestamp_column,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "earliest_timestamp": timestamps.min(),
        "latest_timestamp": timestamps.max(),
    }


def validate_group_split(df, config):
    group_column = config["group_column"]

    if group_column not in df.columns:
        raise ValueError(
            f"Group column {group_column!r} was not found."
        )

    if df[group_column].isnull().any():
        raise ValueError(
            "Group column contains missing values."
        )

    unique_groups = int(df[group_column].nunique())

    if unique_groups < 2:
        raise ValueError(
            "Group-based splitting requires at least two unique groups."
        )

    return {
        "split_type": "group",
        "valid": True,
        "group_column": group_column,
        "unique_groups": unique_groups,
    }