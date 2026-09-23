import pandas as pd
import pytest

from modules.temporal_detector import detect_temporal_leakage


def test_clean_time_split():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        "target": [0, 1, 0],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-04",
            "2026-01-05",
        ],
        "target": [1, 0],
    })

    result = detect_temporal_leakage(
        train_df,
        test_df,
        "date",
        "time",
    )

    assert result["temporal_overlap"] is False
    assert result["risk_detected"] is False
    assert result["configuration_risk"] is False
    assert result["severity"] == "none"

    assert result["latest_train_timestamp"] == pd.Timestamp(
        "2026-01-03"
    )

    assert result["earliest_test_timestamp"] == pd.Timestamp(
        "2026-01-04"
    )


def test_random_split_with_temporal_overlap():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-05",
        ],
        "target": [0, 1, 0],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-03",
            "2026-01-04",
        ],
        "target": [1, 0],
    })

    result = detect_temporal_leakage(
        train_df,
        test_df,
        "date",
        "random",
    )

    assert result["temporal_overlap"] is True
    assert result["risk_detected"] is True
    assert result["configuration_risk"] is True
    assert result["severity"] == "high"


def test_random_split_without_temporal_overlap():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
        ],
        "target": [0, 1],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-03",
            "2026-01-04",
        ],
        "target": [0, 1],
    })

    result = detect_temporal_leakage(
        train_df,
        test_df,
        "date",
        "random",
    )

    assert result["temporal_overlap"] is False
    assert result["risk_detected"] is False
    assert result["configuration_risk"] is True
    assert result["severity"] == "none"
    assert result["recommendation"] is not None


def test_group_split_with_temporal_overlap():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-05",
        ],
        "customer_id": [101, 103],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-02",
            "2026-01-03",
        ],
        "customer_id": [102, 104],
    })

    result = detect_temporal_leakage(
        train_df,
        test_df,
        "date",
        "group",
    )

    assert result["temporal_overlap"] is True
    assert result["configuration_risk"] is True
    assert result["severity"] == "high"


def test_same_timestamp_detected_as_overlap():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-03",
        ],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-03",
            "2026-01-04",
        ],
    })

    result = detect_temporal_leakage(
        train_df,
        test_df,
        "date",
        "time",
    )

    assert result["temporal_overlap"] is True
    assert result["risk_detected"] is True
    assert result["severity"] == "high"


def test_missing_timestamp_column():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3],
    })

    test_df = pd.DataFrame({
        "feature": [4, 5],
    })

    with pytest.raises(
        ValueError,
        match="Timestamp column",
    ):
        detect_temporal_leakage(
            train_df,
            test_df,
            "date",
            "random",
        )


def test_invalid_training_timestamp():
    train_df = pd.DataFrame({
        "date": [
            "2026-01-01",
            "not-a-date",
        ],
    })

    test_df = pd.DataFrame({
        "date": [
            "2026-01-03",
            "2026-01-04",
        ],
    })

    with pytest.raises(
        ValueError,
        match="Training data contains missing or invalid timestamps",
    ):
        detect_temporal_leakage(
            train_df,
            test_df,
            "date",
            "random",
        )


def test_empty_training_data():
    train_df = pd.DataFrame(
        columns=["date"]
    )

    test_df = pd.DataFrame({
        "date": ["2026-01-04"],
    })

    with pytest.raises(
        ValueError,
        match="Training data is empty",
    ):
        detect_temporal_leakage(
            train_df,
            test_df,
            "date",
            "random",
        )


def test_unsupported_split_type():
    train_df = pd.DataFrame({
        "date": ["2026-01-01"],
    })

    test_df = pd.DataFrame({
        "date": ["2026-01-02"],
    })

    with pytest.raises(
        ValueError,
        match="Unsupported split type",
    ):
        detect_temporal_leakage(
            train_df,
            test_df,
            "date",
            "unsupported",
        )