import pandas as pd
import pytest

from modules.proxy_detector import (
    analyze_numerical_target_correlation,
    identify_proxy_candidates,
)

def test_detects_strong_positive_correlation():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "target": [1, 2, 3, 4, 5],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    correlation = result["correlations"][0]

    assert correlation["feature"] == "feature"
    assert correlation["correlation"] == pytest.approx(1.0)
    assert correlation["absolute_correlation"] == pytest.approx(1.0)


def test_detects_strong_negative_correlation():
    df = pd.DataFrame({
        "feature": [5, 4, 3, 2, 1],
        "target": [1, 2, 3, 4, 5],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    correlation = result["correlations"][0]

    assert correlation["correlation"] == pytest.approx(-1.0)
    assert correlation["absolute_correlation"] == pytest.approx(1.0)


def test_multiple_features_are_sorted_by_absolute_correlation():
    df = pd.DataFrame({
        "strong_feature": [1, 2, 3, 4, 5, 6],
        "weaker_feature": [2, 1, 4, 3, 6, 5],
        "target": [1, 2, 3, 4, 5, 6],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    assert (
        result["correlations"][0]["feature"]
        == "strong_feature"
    )

    assert (
        result["correlations"][0]["absolute_correlation"]
        >= result["correlations"][1]["absolute_correlation"]
    )


def test_target_column_is_excluded_from_features():
    df = pd.DataFrame({
        "feature": [10, 20, 30, 40],
        "target": [0, 1, 0, 1],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    feature_names = [
        item["feature"]
        for item in result["correlations"]
    ]

    assert "target" not in feature_names
    assert result["numerical_feature_count"] == 1


def test_non_numerical_features_are_ignored():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "city": [
            "Fullerton",
            "Anaheim",
            "Irvine",
            "Fullerton",
        ],
        "target": [0, 1, 0, 1],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    feature_names = [
        item["feature"]
        for item in result["correlations"]
    ]

    assert "age" in feature_names
    assert "city" not in feature_names
    assert result["numerical_feature_count"] == 1


def test_missing_values_are_ignored_for_correlation():
    df = pd.DataFrame({
        "feature": [1, 2, None, 4, 5],
        "target": [1, 2, 3, 4, 5],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    correlation = result["correlations"][0]

    assert correlation["correlation"] == pytest.approx(1.0)


def test_constant_feature_returns_no_correlation():
    df = pd.DataFrame({
        "feature": [10, 10, 10, 10],
        "target": [0, 1, 0, 1],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    correlation = result["correlations"][0]

    assert correlation["correlation"] is None
    assert correlation["absolute_correlation"] is None


def test_constant_target_returns_no_correlation():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [1, 1, 1, 1],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    correlation = result["correlations"][0]

    assert correlation["correlation"] is None
    assert correlation["absolute_correlation"] is None


def test_returns_analysis_metadata():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [40000, 50000, 60000, 70000],
        "target": [0, 1, 0, 1],
    })

    result = analyze_numerical_target_correlation(
        df,
        "target",
    )

    assert (
        result["analysis_type"]
        == "numerical_target_correlation"
    )
    assert result["target_column"] == "target"
    assert result["numerical_feature_count"] == 2


def test_target_column_is_required():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    })

    with pytest.raises(
        ValueError,
        match="A target column is required",
    ):
        analyze_numerical_target_correlation(
            df,
            None,
        )


def test_target_column_must_exist():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
    })

    with pytest.raises(
        ValueError,
        match="Target column",
    ):
        analyze_numerical_target_correlation(
            df,
            "target",
        )


def test_target_must_be_numerical():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": ["yes", "no", "yes"],
    })

    with pytest.raises(
        ValueError,
        match="requires a numerical target column",
    ):
        analyze_numerical_target_correlation(
            df,
            "target",
        )


def test_empty_dataset():
    df = pd.DataFrame(
        columns=["feature", "target"]
    )

    with pytest.raises(
        ValueError,
        match="Dataset is empty",
    ):
        analyze_numerical_target_correlation(
            df,
            "target",
        )


def test_invalid_input_type():
    df = [1, 2, 3]

    with pytest.raises(
        ValueError,
        match="Input data must be a pandas DataFrame",
    ):
        analyze_numerical_target_correlation(
            df,
            "target",
        )

def test_identifies_strong_proxy_candidate():
    correlation_result = {
        "correlations": [
            {
                "feature": "target_proxy",
                "correlation": 0.98,
                "absolute_correlation": 0.98,
            },
            {
                "feature": "age",
                "correlation": 0.25,
                "absolute_correlation": 0.25,
            },
        ]
    }

    result = identify_proxy_candidates(
        correlation_result
    )

    assert result["candidate_count"] == 1
    assert (
        result["candidates"][0]["feature"]
        == "target_proxy"
    )


def test_identifies_strong_negative_proxy_candidate():
    correlation_result = {
        "correlations": [
            {
                "feature": "inverse_target_proxy",
                "correlation": -0.96,
                "absolute_correlation": 0.96,
            }
        ]
    }

    result = identify_proxy_candidates(
        correlation_result
    )

    assert result["candidate_count"] == 1
    assert (
        result["candidates"][0]["feature"]
        == "inverse_target_proxy"
    )


def test_weak_correlation_is_not_candidate():
    correlation_result = {
        "correlations": [
            {
                "feature": "age",
                "correlation": 0.45,
                "absolute_correlation": 0.45,
            }
        ]
    }

    result = identify_proxy_candidates(
        correlation_result
    )

    assert result["candidate_count"] == 0
    assert result["candidates"] == []


def test_custom_correlation_threshold():
    correlation_result = {
        "correlations": [
            {
                "feature": "feature",
                "correlation": 0.75,
                "absolute_correlation": 0.75,
            }
        ]
    }

    result = identify_proxy_candidates(
        correlation_result,
        correlation_threshold=0.7,
    )

    assert result["candidate_count"] == 1


def test_invalid_correlation_threshold():
    correlation_result = {
        "correlations": []
    }

    with pytest.raises(
        ValueError,
        match="Correlation threshold",
    ):
        identify_proxy_candidates(
            correlation_result,
            correlation_threshold=1.5,
        )


def test_invalid_correlation_result():
    with pytest.raises(
        ValueError,
        match="Correlation result must be a dictionary",
    ):
        identify_proxy_candidates(
            ["invalid"]
        )


def test_missing_correlations_in_result():
    with pytest.raises(
        ValueError,
        match="does not contain correlations",
    ):
        identify_proxy_candidates({})