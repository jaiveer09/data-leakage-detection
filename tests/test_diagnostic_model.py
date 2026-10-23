import pandas as pd
import pytest

from modules.diagnostic_model import (
    analyze_feature_importance,
    compare_clean_and_leaked_models,
    determine_task_type,
    prepare_numerical_features,
    run_diagnostic_model,
)


def test_numeric_binary_target_is_classification():
    target = pd.Series([0, 1, 0, 1])

    result = determine_task_type(target)

    assert result == "classification"


def test_categorical_target_is_classification():
    target = pd.Series([
        "yes",
        "no",
        "yes",
        "no",
    ])

    result = determine_task_type(target)

    assert result == "classification"


def test_continuous_numeric_target_is_regression():
    target = pd.Series([
        float(value)
        for value in range(30)
    ])

    result = determine_task_type(target)

    assert result == "regression"


def test_target_must_be_series():
    with pytest.raises(
        ValueError,
        match="pandas Series",
    ):
        determine_task_type([0, 1, 0, 1])


def test_empty_target_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        determine_task_type(
            pd.Series([], dtype=float)
        )


def test_missing_target_values_rejected():
    target = pd.Series([0, 1, None, 1])

    with pytest.raises(
        ValueError,
        match="cannot contain missing values",
    ):
        determine_task_type(target)


def test_single_value_target_rejected():
    target = pd.Series([1, 1, 1, 1])

    with pytest.raises(
        ValueError,
        match="at least two unique values",
    ):
        determine_task_type(target)


def test_prepare_numerical_features():
    train_df = pd.DataFrame({
        "numeric_1": [1, 2, 3],
        "numeric_2": [4.0, 5.0, 6.0],
        "category": ["a", "b", "c"],
        "target": [0, 1, 0],
    })

    test_df = train_df.copy()

    result = prepare_numerical_features(
        train_df,
        test_df,
        target_column="target",
    )

    assert result == [
        "numeric_1",
        "numeric_2",
    ]


def test_excluded_columns_removed_from_features():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3],
        "group_id": [10, 20, 30],
        "target": [0, 1, 0],
    })

    test_df = train_df.copy()

    result = prepare_numerical_features(
        train_df,
        test_df,
        target_column="target",
        excluded_columns=["group_id"],
    )

    assert result == ["feature"]


def test_no_numerical_features_rejected():
    train_df = pd.DataFrame({
        "category": ["a", "b", "c"],
        "target": [0, 1, 0],
    })

    test_df = train_df.copy()

    with pytest.raises(
        ValueError,
        match="No numerical feature columns",
    ):
        prepare_numerical_features(
            train_df,
            test_df,
            target_column="target",
        )


def test_missing_training_target_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3],
    })

    test_df = pd.DataFrame({
        "feature": [4, 5],
        "target": [0, 1],
    })

    with pytest.raises(
        ValueError,
        match="training dataset",
    ):
        prepare_numerical_features(
            train_df,
            test_df,
            target_column="target",
        )


def test_missing_testing_target_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    })

    test_df = pd.DataFrame({
        "feature": [4, 5],
    })

    with pytest.raises(
        ValueError,
        match="testing dataset",
    ):
        prepare_numerical_features(
            train_df,
            test_df,
            target_column="target",
        )


def test_classification_diagnostic_model():
    train_df = pd.DataFrame({
        "feature": [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        "target": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
    })

    test_df = pd.DataFrame({
        "feature": [2, 3, 6, 7],
        "target": [0, 0, 1, 1],
    })

    result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    assert result["analysis_type"] == "diagnostic_model"
    assert result["task_type"] == "classification"
    assert result["model_name"] == "Logistic Regression"
    assert result["metric_name"] == "accuracy"
    assert result["score"] == 1.0
    assert result["feature_columns"] == ["feature"]
    assert result["feature_count"] == 1


def test_regression_diagnostic_model():
    train_df = pd.DataFrame({
        "feature": list(range(1, 25)),
        "target": [
            (2 * value) + 1
            for value in range(1, 25)
        ],
    })

    test_df = pd.DataFrame({
        "feature": list(range(25, 31)),
        "target": [
            (2 * value) + 1
            for value in range(25, 31)
        ],
    })

    result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    assert result["task_type"] == "regression"
    assert result["model_name"] == "Linear Regression"
    assert result["metric_name"] == "r2"
    assert result["score"] == pytest.approx(1.0)


def test_diagnostic_model_handles_missing_features():
    train_df = pd.DataFrame({
        "feature": [
            1, 2, None, 4,
            5, 6, 7, 8,
        ],
        "target": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
    })

    test_df = pd.DataFrame({
        "feature": [2, None, 6, 7],
        "target": [0, 0, 1, 1],
    })

    result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    assert 0.0 <= result["score"] <= 1.0


def test_testing_target_missing_values_rejected():
    train_df = pd.DataFrame({
        "feature": [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        "target": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
    })

    test_df = pd.DataFrame({
        "feature": [2, 3, 6, 7],
        "target": [0, None, 1, 1],
    })

    with pytest.raises(
        ValueError,
        match="Testing target data cannot contain missing values",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )


@pytest.mark.parametrize(
    "train_data,test_data,error_message",
    [
        (
            [],
            pd.DataFrame({
                "feature": [1],
                "target": [0],
            }),
            "Training data must be a pandas DataFrame",
        ),
        (
            pd.DataFrame({
                "feature": [1],
                "target": [0],
            }),
            [],
            "Testing data must be a pandas DataFrame",
        ),
    ],
)
def test_diagnostic_model_requires_dataframes(
    train_data,
    test_data,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        run_diagnostic_model(
            train_data,
            test_data,
            target_column="target",
        )


def test_empty_training_data_rejected():
    train_df = pd.DataFrame(
        columns=["feature", "target"]
    )

    test_df = pd.DataFrame({
        "feature": [1, 2],
        "target": [0, 1],
    })

    with pytest.raises(
        ValueError,
        match="Training data cannot be empty",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )


def test_empty_testing_data_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2],
        "target": [0, 1],
    })

    test_df = pd.DataFrame(
        columns=["feature", "target"]
    )

    with pytest.raises(
        ValueError,
        match="Testing data cannot be empty",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )

def test_clean_leaked_comparison_detects_inflation():
    train_df = pd.DataFrame({
        "normal_feature": [
            4, 8, 2, 9, 5, 1, 7, 3,
            6, 8, 2, 5, 9, 1, 4, 7,
            3, 6, 5, 8, 1, 9, 2, 7,
        ],
        "leaked_feature": [
            0, 0, 1, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 0, 1,
            0, 1, 1, 0, 0, 1, 0, 1,
        ],
        "target": [
            0, 0, 1, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 0, 1,
            0, 1, 1, 0, 0, 1, 0, 1,
        ],
    })

    test_df = pd.DataFrame({
        "normal_feature": [
            5, 2, 8, 4, 7, 1,
            9, 3, 6, 5, 2, 8,
        ],
        "leaked_feature": [
            1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0,
        ],
        "target": [
            1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0,
        ],
    })

    result = compare_clean_and_leaked_models(
        train_df,
        test_df,
        target_column="target",
        leaked_feature_columns=["leaked_feature"],
    )

    assert result["analysis_type"] == (
        "clean_leaked_comparison"
    )
    assert result["task_type"] == "classification"
    assert result["metric_name"] == "accuracy"

    assert result["clean_score"] == 0.75
    assert result["leaked_score"] == 1.0
    assert result["score_difference"] == 0.25
    assert result["performance_inflated"] is True

    assert result["clean_feature_columns"] == [
        "normal_feature"
    ]

    assert result[
        "leaked_model_feature_columns"
    ] == [
        "normal_feature",
        "leaked_feature",
    ]


def test_clean_leaked_comparison_metadata():
    train_df = pd.DataFrame({
        "feature": [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        "leak": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
        "target": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
    })

    test_df = pd.DataFrame({
        "feature": [2, 3, 6, 7],
        "leak": [0, 0, 1, 1],
        "target": [0, 0, 1, 1],
    })

    result = compare_clean_and_leaked_models(
        train_df,
        test_df,
        target_column="target",
        leaked_feature_columns=["leak"],
    )

    assert result["leaked_feature_columns"] == [
        "leak"
    ]

    assert "clean_result" in result
    assert "leaked_result" in result


def test_clean_leaked_comparison_requires_list():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "leak": [0, 0, 1, 1],
        "target": [0, 0, 1, 1],
    })

    with pytest.raises(
        ValueError,
        match="must be provided as a list",
    ):
        compare_clean_and_leaked_models(
            train_df,
            train_df.copy(),
            target_column="target",
            leaked_feature_columns="leak",
        )


def test_clean_leaked_comparison_requires_feature():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 1, 1],
    })

    with pytest.raises(
        ValueError,
        match="At least one leaked feature",
    ):
        compare_clean_and_leaked_models(
            train_df,
            train_df.copy(),
            target_column="target",
            leaked_feature_columns=[],
        )


def test_missing_leaked_feature_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 1, 1],
    })

    test_df = train_df.copy()

    with pytest.raises(
        ValueError,
        match="not found in both",
    ):
        compare_clean_and_leaked_models(
            train_df,
            test_df,
            target_column="target",
            leaked_feature_columns=[
                "missing_leak"
            ],
        )


def test_existing_exclusions_preserved():
    train_df = pd.DataFrame({
        "feature": [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        "group_id": [
            10, 10, 20, 20,
            30, 30, 40, 40,
        ],
        "leak": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
        "target": [
            0, 0, 0, 0,
            1, 1, 1, 1,
        ],
    })

    test_df = pd.DataFrame({
        "feature": [2, 3, 6, 7],
        "group_id": [10, 20, 30, 40],
        "leak": [0, 0, 1, 1],
        "target": [0, 0, 1, 1],
    })

    result = compare_clean_and_leaked_models(
        train_df,
        test_df,
        target_column="target",
        leaked_feature_columns=["leak"],
        excluded_columns=["group_id"],
    )

    assert "group_id" not in result[
        "clean_feature_columns"
    ]

    assert "group_id" not in result[
        "leaked_model_feature_columns"
    ]

    assert "leak" not in result[
        "clean_feature_columns"
    ]

    assert "leak" in result[
        "leaked_model_feature_columns"
    ]

def test_feature_importance_identifies_top_feature():
    train_df = pd.DataFrame({
        "strong_feature": list(range(1, 25)),
        "weak_feature": [
            4, 1, 7, 2, 8, 3, 6, 5,
            2, 8, 1, 7, 4, 6, 3, 5,
            8, 2, 6, 1, 5, 3, 7, 4,
        ],
        "target": [
            (10 * value) + 5
            for value in range(1, 25)
        ],
    })

    test_df = pd.DataFrame({
        "strong_feature": list(range(25, 31)),
        "weak_feature": [3, 7, 1, 6, 2, 5],
        "target": [
            (10 * value) + 5
            for value in range(25, 31)
        ],
    })

    diagnostic_result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    result = analyze_feature_importance(
        diagnostic_result
    )

    assert result["analysis_type"] == (
        "feature_importance"
    )
    assert result["task_type"] == "regression"
    assert result["feature_count"] == 2
    assert result["top_feature"] == (
        "strong_feature"
    )

    assert (
        result["feature_importance"][0][
            "importance"
        ]
        >
        result["feature_importance"][1][
            "importance"
        ]
    )


def test_feature_importance_requires_dictionary():
    with pytest.raises(
        ValueError,
        match="must be a dictionary",
    ):
        analyze_feature_importance([])


def test_feature_importance_requires_fields():
    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        analyze_feature_importance({})

def test_comparison_uses_identical_train_test_rows_and_metric(monkeypatch):
    import modules.diagnostic_model as diagnostic_module

    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "suspected_proxy": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    }, index=[10, 20, 30, 40])

    test_df = pd.DataFrame({
        "feature": [5, 6],
        "suspected_proxy": [0, 1],
        "target": [0, 1],
    }, index=[50, 60])

    calls = []

    def fake_run_diagnostic_model(
        train_df,
        test_df,
        target_column,
        excluded_columns=None,
    ):
        calls.append({
            "train_df": train_df.copy(deep=True),
            "test_df": test_df.copy(deep=True),
            "target_column": target_column,
            "excluded_columns": list(excluded_columns or []),
        })

        return {
            "task_type": "classification",
            "metric_name": "accuracy",
            "score": 0.7 if "suspected_proxy" in (
                excluded_columns or []
            ) else 0.9,
            "feature_columns": [
                "feature"
            ] if "suspected_proxy" in (
                excluded_columns or []
            ) else ["feature", "suspected_proxy"],
        }

    monkeypatch.setattr(
        diagnostic_module,
        "run_diagnostic_model",
        fake_run_diagnostic_model,
    )

    result = compare_clean_and_leaked_models(
        train_df,
        test_df,
        target_column="target",
        leaked_feature_columns=["suspected_proxy"],
    )

    assert len(calls) == 2

    for call in calls:
        pd.testing.assert_frame_equal(
            call["train_df"],
            train_df,
        )
        pd.testing.assert_frame_equal(
            call["test_df"],
            test_df,
        )
        assert call["target_column"] == "target"

    assert "suspected_proxy" in calls[0]["excluded_columns"]
    assert "suspected_proxy" not in calls[1]["excluded_columns"]

    assert result["metric_name"] == "accuracy"
    assert result["clean_score"] == pytest.approx(0.7)
    assert result["leaked_score"] == pytest.approx(0.9)
    assert result["score_difference"] == pytest.approx(0.2)
    
    
def test_diagnostic_model_with_constant_feature():
    train_df = pd.DataFrame({
        "constant_feature": [5] * 8,
        "useful_feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "constant_feature": [5] * 4,
        "useful_feature": [2, 3, 6, 7],
        "target": [0, 0, 1, 1],
    })

    result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    assert result["task_type"] == "classification"
    assert 0.0 <= result["score"] <= 1.0
    assert result["feature_count"] == 2


def test_constant_feature_has_zero_importance():
    train_df = pd.DataFrame({
        "constant_feature": [5] * 8,
        "useful_feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "constant_feature": [5] * 4,
        "useful_feature": [2, 3, 6, 7],
        "target": [0, 0, 1, 1],
    })

    diagnostic_result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    importance_result = analyze_feature_importance(
        diagnostic_result
    )

    importance_by_feature = {
        item["feature"]: item["importance"]
        for item in importance_result["feature_importance"]
    }

    assert importance_by_feature["constant_feature"] == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_diagnostic_model_with_only_constant_features():
    train_df = pd.DataFrame({
        "constant_feature": [5] * 8,
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "constant_feature": [5] * 4,
        "target": [0, 1, 0, 1],
    })

    result = run_diagnostic_model(
        train_df,
        test_df,
        target_column="target",
    )

    assert result["task_type"] == "classification"
    assert 0.0 <= result["score"] <= 1.0
    assert result["feature_columns"] == ["constant_feature"]


def test_constant_target_is_rejected_by_diagnostic_model():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "target": [1, 1, 1, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "feature": [7, 8],
        "target": [1, 1],
    })

    with pytest.raises(
        ValueError,
        match="at least two unique values",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )
        


def test_all_missing_numerical_feature_rejected():
    train_df = pd.DataFrame({
        "feature": [float("nan")] * 8,
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "feature": [float("nan")] * 4,
        "target": [0, 0, 1, 1],
    })

    with pytest.raises(
        ValueError,
        match="no non-missing training values",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )



def test_missing_training_target_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "target": [0, 0, None, 1, 1, 1],
    })

    test_df = pd.DataFrame({
        "feature": [7, 8],
        "target": [0, 1],
    })

    with pytest.raises(
        ValueError,
        match="Target data cannot contain missing values",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )


def test_single_class_training_split_rejected():
    train_df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 0, 0],
    })

    test_df = pd.DataFrame({
        "feature": [5, 6],
        "target": [0, 1],
    })

    with pytest.raises(
        ValueError,
        match="at least two unique values",
    ):
        run_diagnostic_model(
            train_df,
            test_df,
            target_column="target",
        )

