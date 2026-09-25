import pandas as pd


def detect_group_leakage(
    train_df,
    test_df,
    group_column,
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

    if not group_column:
        raise ValueError("A group column is required.")

    if group_column not in train_df.columns:
        raise ValueError(
            f"Group column {group_column!r} "
            "was not found in the training data."
        )

    if group_column not in test_df.columns:
        raise ValueError(
            f"Group column {group_column!r} "
            "was not found in the testing data."
        )

    if train_df[group_column].isnull().any():
        raise ValueError(
            "Training data contains missing group values."
        )

    if test_df[group_column].isnull().any():
        raise ValueError(
            "Testing data contains missing group values."
        )

    train_groups = set(
        train_df[group_column].unique()
    )

    test_groups = set(
        test_df[group_column].unique()
    )

    overlapping_groups = train_groups.intersection(
        test_groups
    )

    overlap_count = len(overlapping_groups)

    leakage_detected = overlap_count > 0

    if leakage_detected:
        severity = "high"
        message = (
            "Group leakage detected. One or more group values "
            "appear in both the training and testing sets."
        )
        recommendation = (
            "Use a group-aware splitting strategy so that each "
            "group appears in only one of the training or testing sets."
        )
    else:
        severity = "none"
        message = (
            "No group leakage detected. Training and testing "
            "sets contain separate group values."
        )
        recommendation = None

    return {
        "leakage_type": "group",
        "group_column": group_column,
        "leakage_detected": leakage_detected,
        "risk_detected": leakage_detected,
        "severity": severity,
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "overlap_count": overlap_count,
        "overlapping_groups": [
            value.item() if hasattr(value, "item") else value
            for value in sorted(overlapping_groups, key=str)
        ],
        "message": message,
        "recommendation": recommendation,
    }