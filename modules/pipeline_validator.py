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