def validate_preprocessing_configuration(
    preprocessing_steps,
):
    if preprocessing_steps is None:
        preprocessing_steps = []

    if not isinstance(preprocessing_steps, list):
        raise ValueError(
            "Preprocessing steps must be provided as a list."
        )

    findings = []

    for index, step in enumerate(preprocessing_steps):
        if not isinstance(step, dict):
            raise ValueError(
                "Each preprocessing step must be a dictionary."
            )

        name = step.get("name")

        if not name:
            raise ValueError(
                f"Preprocessing step {index + 1} requires a name."
            )

        fit_scope = step.get("fit_scope")

        if fit_scope not in {"training", "full_dataset"}:
            raise ValueError(
                f"Preprocessing step {name!r} must specify "
                "fit_scope as 'training' or 'full_dataset'."
            )

        if fit_scope == "full_dataset":
            findings.append({
                "step": name,
                "risk_detected": True,
                "severity": "high",
                "message": (
                    f"Preprocessing step {name!r} is configured "
                    "to learn from the full dataset before "
                    "evaluation."
                ),
                "recommendation": (
                    "Fit this preprocessing step using the "
                    "training data only, then apply the fitted "
                    "transformation to the testing data."
                ),
            })
        else:
            findings.append({
                "step": name,
                "risk_detected": False,
                "severity": "none",
                "message": (
                    f"Preprocessing step {name!r} is configured "
                    "to learn from training data only."
                ),
                "recommendation": None,
            })

    risky_steps = [
        finding
        for finding in findings
        if finding["risk_detected"]
    ]

    risk_detected = len(risky_steps) > 0

    if not preprocessing_steps:
        summary = (
            "No preprocessing steps were provided for validation."
        )
    elif risk_detected:
        summary = (
            f"{len(risky_steps)} preprocessing step(s) may "
            "introduce data leakage."
        )
    else:
        summary = (
            "No preprocessing leakage risk was identified "
            "in the provided configuration."
        )

    return {
        "analysis_type": "preprocessing_validation",
        "step_count": len(preprocessing_steps),
        "risky_step_count": len(risky_steps),
        "risk_detected": risk_detected,
        "severity": "high" if risk_detected else "none",
        "findings": findings,
        "summary": summary,
    }

def validate_cross_validation_configuration(
    split_type,
    cv_strategy,
    cv_folds,
):
    valid_split_types = {
        "random",
        "time",
        "group",
    }

    valid_cv_strategies = {
        "kfold",
        "time_series",
        "group",
    }

    if split_type not in valid_split_types:
        raise ValueError(
            f"Unsupported split type: {split_type!r}."
        )

    if cv_strategy not in valid_cv_strategies:
        raise ValueError(
            f"Unsupported cross-validation strategy: "
            f"{cv_strategy!r}."
        )

    if (
        not isinstance(cv_folds, int)
        or isinstance(cv_folds, bool)
        or cv_folds < 2
    ):
        raise ValueError(
            "cv_folds must be an integer of at least 2."
        )

    recommended_strategy = {
        "random": "kfold",
        "time": "time_series",
        "group": "group",
    }[split_type]

    risk_detected = (
        cv_strategy != recommended_strategy
    )

    if risk_detected:
        severity = "high"

        if split_type == "time":
            message = (
                "The selected cross-validation strategy does "
                "not preserve chronological ordering for "
                "time-based data."
            )
            recommendation = (
                "Use time-series cross-validation so that "
                "validation observations occur after the "
                "training observations."
            )

        elif split_type == "group":
            message = (
                "The selected cross-validation strategy does "
                "not guarantee separation between groups."
            )
            recommendation = (
                "Use group-aware cross-validation so that "
                "records from the same group are not divided "
                "between training and validation folds."
            )

        else:
            message = (
                "The selected cross-validation strategy does "
                "not match the random evaluation configuration."
            )
            recommendation = (
                "Use standard K-fold cross-validation for the "
                "current random evaluation configuration, "
                "unless the dataset requires temporal or group "
                "separation."
            )

    else:
        severity = "none"

        if split_type == "time":
            message = (
                "The selected time-series cross-validation "
                "strategy is appropriate for preserving "
                "chronological ordering."
            )

        elif split_type == "group":
            message = (
                "The selected group-aware cross-validation "
                "strategy is appropriate for preserving "
                "group separation."
            )

        else:
            message = (
                "The selected K-fold cross-validation strategy "
                "is appropriate for the random evaluation "
                "configuration."
            )

        recommendation = None

    return {
        "analysis_type": "cross_validation_validation",
        "split_type": split_type,
        "cv_strategy": cv_strategy,
        "cv_folds": cv_folds,
        "recommended_strategy": recommended_strategy,
        "risk_detected": risk_detected,
        "severity": severity,
        "message": message,
        "recommendation": recommendation,
    }