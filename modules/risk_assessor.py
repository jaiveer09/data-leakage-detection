VALID_SEVERITIES = {
    "none",
    "low",
    "medium",
    "high",
}


def create_leakage_finding(
    category,
    title,
    message,
    severity,
    risk_detected,
    recommendation=None,
    details=None,
):
    if not isinstance(category, str) or not category.strip():
        raise ValueError(
            "Finding category must be a non-empty string."
        )

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Finding title must be a non-empty string."
        )

    if not isinstance(message, str) or not message.strip():
        raise ValueError(
            "Finding message must be a non-empty string."
        )

    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Unsupported finding severity: {severity!r}."
        )

    if not isinstance(risk_detected, bool):
        raise ValueError(
            "risk_detected must be a boolean value."
        )

    if risk_detected and severity == "none":
        raise ValueError(
            "A detected risk cannot have severity 'none'."
        )

    if not risk_detected and severity != "none":
        raise ValueError(
            "A finding without detected risk must have "
            "severity 'none'."
        )

    if (
        recommendation is not None
        and not isinstance(recommendation, str)
    ):
        raise ValueError(
            "Recommendation must be a string or None."
        )

    if details is None:
        details = {}

    if not isinstance(details, dict):
        raise ValueError(
            "Finding details must be a dictionary."
        )

    return {
        "category": category.strip(),
        "title": title.strip(),
        "message": message.strip(),
        "severity": severity,
        "risk_detected": risk_detected,
        "recommendation": recommendation,
        "details": details,
    }


def assess_overall_risk(findings):
    if not isinstance(findings, list):
        raise ValueError(
            "Findings must be provided as a list."
        )

    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError(
                "Each finding must be a dictionary."
            )

        required_fields = {
            "category",
            "title",
            "message",
            "severity",
            "risk_detected",
        }

        missing_fields = (
            required_fields - finding.keys()
        )

        if missing_fields:
            raise ValueError(
                "Finding is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        if finding["severity"] not in VALID_SEVERITIES:
            raise ValueError(
                "Finding contains an unsupported severity."
            )

        if not isinstance(
            finding["risk_detected"],
            bool,
        ):
            raise ValueError(
                "Finding risk_detected value must be boolean."
            )

        if (
            finding["risk_detected"]
            and finding["severity"] == "none"
        ):
            raise ValueError(
                "Detected risks cannot have severity 'none'."
            )

        if (
            not finding["risk_detected"]
            and finding["severity"] != "none"
        ):
            raise ValueError(
                "Findings without detected risk must have "
                "severity 'none'."
            )

    risky_findings = [
        finding
        for finding in findings
        if finding["risk_detected"]
    ]

    severity_rank = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    if risky_findings:
        overall_risk = max(
            (
                finding["severity"]
                for finding in risky_findings
            ),
            key=lambda severity: severity_rank[severity],
        )
    else:
        overall_risk = "none"

    severity_counts = {
        "low": 0,
        "medium": 0,
        "high": 0,
    }

    for finding in risky_findings:
        severity_counts[finding["severity"]] += 1

    if overall_risk == "none":
        summary = (
            "No data leakage risks were identified "
            "in the analyzed findings."
        )
    else:
        summary = (
            f"{len(risky_findings)} data leakage risk(s) "
            f"were identified. Overall risk: "
            f"{overall_risk}."
        )

    return {
        "analysis_type": "overall_risk_assessment",
        "overall_risk": overall_risk,
        "risk_detected": len(risky_findings) > 0,
        "finding_count": len(findings),
        "risky_finding_count": len(risky_findings),
        "severity_counts": severity_counts,
        "findings": findings,
        "summary": summary,
    }