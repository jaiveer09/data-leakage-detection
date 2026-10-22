
import json

import pytest

from modules.report_generator import (
    build_analysis_report,
    report_to_json,
)


def make_report(**overrides):
    args = {
        "dataset_name": "sample.csv",
        "profile": {"rows": 10, "columns": 3},
        "config": {
            "target_column": "target",
            "split_type": "random",
            "test_size": 0.2,
            "cv_folds": 5,
        },
        "cv_strategy": "kfold",
        "preprocessing_steps": [],
        "overall_risk_result": {
            "overall_risk": "medium",
            "summary": "One potential risk.",
            "finding_count": 1,
            "risky_finding_count": 1,
            "severity_counts": {
                "low": 0,
                "medium": 1,
                "high": 0,
            },
            "findings": [
                {
                    "category": "Proxy Leakage",
                    "severity": "medium",
                    "risk_detected": True,
                }
            ],
        },
    }

    args.update(overrides)
    return build_analysis_report(**args)


def test_report_includes_dataset_config_and_risks():
    report = make_report()

    assert report["dataset"]["name"] == "sample.csv"
    assert report["evaluation"]["split_type"] == "random"
    assert report["risk_assessment"]["overall_risk"] == "medium"
    assert len(report["risk_assessment"]["findings"]) == 1
    assert report["model_comparison"]["status"] == "not_requested"
    assert report["diagnostic_model"]["status"] == "unavailable"


def test_report_excludes_model_pipeline_and_includes_scores():
    report = make_report(
        diagnostic_result={
            "task_type": "classification",
            "model_name": "Logistic Regression",
            "metric_name": "accuracy",
            "score": 0.9,
            "feature_columns": ["income"],
            "train_rows": 8,
            "test_rows": 2,
            "pipeline": object(),
        },
        comparison_result={
            "leaked_feature_columns": ["proxy"],
            "metric_name": "accuracy",
            "clean_score": 0.6,
            "leaked_score": 0.9,
            "score_difference": 0.3,
            "performance_inflated": True,
        },
        selected_leakage_features=["proxy"],
    )

    assert report["diagnostic_model"]["score"] == 0.9
    assert "pipeline" not in report["diagnostic_model"]
    assert report["model_comparison"]["score_difference"] == 0.3

    decoded = json.loads(report_to_json(report))
    assert decoded["risk_assessment"]["overall_risk"] == "medium"


def test_unavailable_comparison_keeps_reason():
    report = make_report(
        selected_leakage_features=["proxy"],
        comparison_error="No predictors",
    )

    assert report["model_comparison"] == {
        "status": "unavailable",
        "reason": "No predictors",
    }


@pytest.mark.parametrize(
    "field",
    [
        "dataset_name",
        "profile",
        "config",
        "overall_risk_result",
        "preprocessing_steps",
    ],
)
def test_invalid_inputs_rejected(field):
    invalid = {
        "dataset_name": "",
        "profile": None,
        "config": None,
        "overall_risk_result": None,
        "preprocessing_steps": None,
    }

    with pytest.raises(ValueError):
        make_report(**{field: invalid[field]})


def test_report_to_json_requires_dict():
    with pytest.raises(ValueError):
        report_to_json([])
