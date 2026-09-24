import pandas as pd


def detect_temporal_leakage(
    train_df,
    test_df,
    timestamp_column,
    split_type,
):
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError(
            "Training data must be a pandas DataFrame."
        )

    if not isinstance(test_df, pd.DataFrame):
        raise ValueError(
            "Testing data must be a pandas DataFrame."
        )

    if train_df.empty:
        raise ValueError("Training data is empty.")

    if test_df.empty:
        raise ValueError("Testing data is empty.")

    if not timestamp_column:
        raise ValueError("A timestamp column is required.")

    if timestamp_column not in train_df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} "
            "was not found in the training data."
        )

    if timestamp_column not in test_df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} "
            "was not found in the testing data."
        )

    if split_type not in {"random", "time", "group"}:
        raise ValueError(
            f"Unsupported split type: {split_type!r}."
        )

    train_timestamps = pd.to_datetime(
        train_df[timestamp_column],
        errors="coerce",
    )

    test_timestamps = pd.to_datetime(
        test_df[timestamp_column],
        errors="coerce",
    )

    if train_timestamps.isna().any():
        raise ValueError(
            "Training data contains missing or invalid timestamps."
        )

    if test_timestamps.isna().any():
        raise ValueError(
            "Testing data contains missing or invalid timestamps."
        )

    earliest_train_timestamp = train_timestamps.min()
    latest_train_timestamp = train_timestamps.max()

    earliest_test_timestamp = test_timestamps.min()
    latest_test_timestamp = test_timestamps.max()

    temporal_overlap = (
        latest_train_timestamp >= earliest_test_timestamp
    )

    if split_type == "time":
        configuration_risk = False
        configuration_message = (
            "The selected time-based split is designed to "
            "preserve chronological ordering."
        )

    elif split_type == "random":
        configuration_risk = True
        configuration_message = (
            "Random splitting does not guarantee chronological "
            "ordering when temporal data is present."
        )

    else:
        configuration_risk = True
        configuration_message = (
            "Group-based splitting preserves group separation "
            "but does not guarantee chronological ordering."
        )

    if temporal_overlap:
        severity = "high"
        message = (
            "Temporal overlap detected. The training set contains "
            "observations that occur at or after the earliest "
            "testing observation."
        )
        if (
            split_type == "time"
            and latest_train_timestamp == earliest_test_timestamp
        ):
            recommendation = (
                "Adjust the time-based split boundary so that "
                "observations with the same timestamp are not divided "
                "between the training and testing sets."
            )
        else:
            recommendation = (
                "Use a splitting strategy that keeps training "
                "observations chronologically before testing "
                "observations."
            )
    else:
        severity = "none"
        message = (
            "No temporal overlap detected in the generated split. "
            "All training observations occur before the earliest "
            "testing observation."
        )

        if configuration_risk:
            recommendation = (
                "The current split has no observed temporal overlap, "
                "but the selected splitting strategy does not "
                "guarantee chronological ordering."
            )
        else:
            recommendation = None

    return {
        "leakage_type": "temporal",
        "split_type": split_type,
        "configuration_risk": configuration_risk,
        "configuration_message": configuration_message,
        "temporal_overlap": temporal_overlap,
        "risk_detected": temporal_overlap,
        "severity": severity,
        "earliest_train_timestamp": earliest_train_timestamp,
        "latest_train_timestamp": latest_train_timestamp,
        "earliest_test_timestamp": earliest_test_timestamp,
        "latest_test_timestamp": latest_test_timestamp,
        "message": message,
        "recommendation": recommendation,
    }
