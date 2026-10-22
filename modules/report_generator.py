
"""Build and serialize a structured data leakage analysis report."""

import json
from datetime import datetime, timezone


def build_analysis_report(
    dataset_name,
    profile,
    config,
    cv_strategy,
    preprocessing_steps,
    overall_risk_result,
    diagnostic_result=None,
    diagnostic_error=None,
    comparison_result=None,
    comparison_error=None,
    selected_leakage_features=None,
):
    """Return a JSON-compatible report without fitted sklearn objects."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("Dataset name must be a non-empty string.")

    for name, value in (
        ("profile", profile),
        ("config", config),
        ("overall_risk_result", overall_risk_result),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a dictionary.")

    if not isinstance(overall_risk_result.get("findings"), list):
        raise ValueError("Risk assessment must contain a findings list.")

    if not isinstance(preprocessing_steps, list):
        raise ValueError("Preprocessing steps must be a list.")

    diagnostic = {"status": "unavailable", "reason": diagnostic_error}

    if diagnostic_result is not None:
        diagnostic = {
            "status": "completed",
            "task_type": diagnostic_result["task_type"],
            "model_name": diagnostic_result["model_name"],
            "metric_name": diagnostic_result["metric_name"],
            "score": diagnostic_result["score"],
            "feature_columns": diagnostic_result["feature_columns"],
            "train_rows": diagnostic_result["train_rows"],
            "test_rows": diagnostic_result["test_rows"],
        }

    comparison = {"status": "not_requested"}

    if selected_leakage_features:
        comparison = {"status": "unavailable", "reason": comparison_error}

        if comparison_result is not None:
            comparison = {
                "status": "completed",
                "suspected_features": comparison_result["leaked_feature_columns"],
                "metric_name": comparison_result["metric_name"],
                "score_without_features": comparison_result["clean_score"],
                "score_with_features": comparison_result["leaked_score"],
                "score_difference": comparison_result["score_difference"],
                "performance_increased": comparison_result["performance_inflated"],
                "interpretation": (
                    "A higher score does not by itself prove leakage. "
                    "Review prediction-time feature availability."
                ),
            }

    return {
        "report_title": "Data Leakage Analysis Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": dataset_name, "profile": profile},
        "evaluation": {**config, "cv_strategy": cv_strategy},
        "preprocessing_steps": preprocessing_steps,
        "risk_assessment": {
            "overall_risk": overall_risk_result["overall_risk"],
            "summary": overall_risk_result["summary"],
            "finding_count": overall_risk_result["finding_count"],
            "risky_finding_count": overall_risk_result["risky_finding_count"],
            "severity_counts": overall_risk_result["severity_counts"],
            "findings": overall_risk_result["findings"],
        },
        "diagnostic_model": diagnostic,
        "model_comparison": comparison,
        "limitations": [
            "Leakage findings indicate potential risks and require contextual review.",
            "A performance increase alone does not prove data leakage.",
            "Preprocessing and cross-validation checks assess configured settings.",
        ],
    }


def report_to_json(report):
    """Serialize a structured report as readable UTF-8 JSON text."""
    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    return json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
