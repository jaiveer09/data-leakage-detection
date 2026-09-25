import pandas as pd
import pytest

from modules.group_detector import detect_group_leakage


def test_no_group_leakage():
    train_df = pd.DataFrame({
        "customer_id": [101, 101, 102, 102],
        "target": [0, 1, 0, 1],
    })

    test_df = pd.DataFrame({
        "customer_id": [103, 103, 104],
        "target": [1, 0, 1],
    })

    result = detect_group_leakage(
        train_df,
        test_df,
        "customer_id",
    )

    assert result["leakage_detected"] is False
    assert result["risk_detected"] is False
    assert result["severity"] == "none"
    assert result["overlap_count"] == 0
    assert result["overlapping_groups"] == []
    assert result["train_group_count"] == 2
    assert result["test_group_count"] == 2


def test_group_leakage_detected():
    train_df = pd.DataFrame({
        "customer_id": [101, 101, 102, 103],
        "target": [0, 1, 0, 1],
    })

    test_df = pd.DataFrame({
        "customer_id": [103, 104, 105],
        "target": [1, 0, 1],
    })

    result = detect_group_leakage(
        train_df,
        test_df,
        "customer_id",
    )

    assert result["leakage_detected"] is True
    assert result["risk_detected"] is True
    assert result["severity"] == "high"
    assert result["overlap_count"] == 1
    assert result["overlapping_groups"] == [103]


def test_multiple_overlapping_groups():
    train_df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
    })

    test_df = pd.DataFrame({
        "customer_id": [102, 104, 105, 106],
    })

    result = detect_group_leakage(
        train_df,
        test_df,
        "customer_id",
    )

    assert result["leakage_detected"] is True
    assert result["overlap_count"] == 2
    assert result["overlapping_groups"] == [102, 104]


def test_string_group_values():
    train_df = pd.DataFrame({
        "patient_id": [
            "patient_a",
            "patient_b",
            "patient_c",
        ],
    })

    test_df = pd.DataFrame({
        "patient_id": [
            "patient_c",
            "patient_d",
        ],
    })

    result = detect_group_leakage(
        train_df,
        test_df,
        "patient_id",
    )

    assert result["leakage_detected"] is True
    assert result["overlap_count"] == 1
    assert result["overlapping_groups"] == ["patient_c"]


def test_group_column_required():
    train_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    test_df = pd.DataFrame({
        "customer_id": [103, 104],
    })

    with pytest.raises(
        ValueError,
        match="A group column is required",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            None,
        )


def test_group_column_missing_from_training_data():
    train_df = pd.DataFrame({
        "feature": [1, 2],
    })

    test_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    with pytest.raises(
        ValueError,
        match="Group column",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_group_column_missing_from_testing_data():
    train_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    test_df = pd.DataFrame({
        "feature": [1, 2],
    })

    with pytest.raises(
        ValueError,
        match="Group column",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_missing_training_group_value():
    train_df = pd.DataFrame({
        "customer_id": [101, None, 103],
    })

    test_df = pd.DataFrame({
        "customer_id": [104, 105],
    })

    with pytest.raises(
        ValueError,
        match="Training data contains missing group values",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_missing_testing_group_value():
    train_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    test_df = pd.DataFrame({
        "customer_id": [103, None],
    })

    with pytest.raises(
        ValueError,
        match="Testing data contains missing group values",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_empty_training_data():
    train_df = pd.DataFrame(
        columns=["customer_id"]
    )

    test_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    with pytest.raises(
        ValueError,
        match="Training data is empty",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_empty_testing_data():
    train_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    test_df = pd.DataFrame(
        columns=["customer_id"]
    )

    with pytest.raises(
        ValueError,
        match="Testing data is empty",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_invalid_training_input():
    train_df = [101, 102]

    test_df = pd.DataFrame({
        "customer_id": [103, 104],
    })

    with pytest.raises(
        ValueError,
        match="Training data must be a pandas DataFrame",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )


def test_invalid_testing_input():
    train_df = pd.DataFrame({
        "customer_id": [101, 102],
    })

    test_df = [103, 104]

    with pytest.raises(
        ValueError,
        match="Testing data must be a pandas DataFrame",
    ):
        detect_group_leakage(
            train_df,
            test_df,
            "customer_id",
        )