VALID_SPLIT_TYPES = {"random", "time", "group"}


def create_config(
    target_column,
    timestamp_column=None,
    group_column=None,
    split_type="random",
    test_size=0.2,
    cv_folds=5,
):
    if not target_column:
        raise ValueError("A target column is required.")

    if split_type not in VALID_SPLIT_TYPES:
        raise ValueError(
            f"Invalid split type: {split_type!r}. "
            f"Choose from {sorted(VALID_SPLIT_TYPES)}."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer of at least 2.")

    if split_type == "time" and not timestamp_column:
        raise ValueError("A timestamp column is required for a time-based split.")

    if split_type == "group" and not group_column:
        raise ValueError("A group column is required for a group-based split.")

    return {
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "group_column": group_column,
        "split_type": split_type,
        "test_size": test_size,
        "cv_folds": cv_folds,
    }