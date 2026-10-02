import pytest

from modules.risk_assessor import (
    assess_overall_risk,
    create_leakage_finding,
)


def test_create_safe_finding():
    result = create_leakage_finding(
        category="temporal",
        title="Temporal Check",
        message="No temporal leakage detected.",
        severity="none",
        risk_detected=False,
    )

    assert result["category"] == "temporal"
    assert result["title"] == "Temporal Check"
    assert result["severity"] == "none"
    assert result["risk_detected"] is False
    assert result["recommendation"] is None
    assert result["details"] == {}


def test_create_risky_finding():
    result = create_leakage_finding(
        category="group",
        title="Group Leakage",
        message="Groups overlap across the split.",
        severity="high",
        risk_detected=True,
        recommendation="Use a group-aware split.",
        details={
            "overlapping_groups": 3,
        },
    )

    assert result["severity"] == "high"
    assert result["risk_detected"] is True
    assert result["recommendation"] == (
        "Use a group-aware split."
    )
    assert result["details"] == {
        "overlapping_groups": 3,
    }


@pytest.mark.parametrize(
    "category,title,message",
    [
        ("", "Title", "Message"),
        ("   ", "Title", "Message"),
        ("group", "", "Message"),
        ("group", "   ", "Message"),
        ("group", "Title", ""),
        ("group", "Title", "   "),
    ],
)
def test_required_finding_text(
    category,
    title,
    message,
):
    with pytest.raises(ValueError):
        create_leakage_finding(
            category=category,
            title=title,
            message=message,
            severity="none",
            risk_detected=False,
        )


def test_invalid_severity_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported finding severity",
    ):
        create_leakage_finding(
            category="proxy",
            title="Proxy Check",
            message="Test message.",
            severity="critical",
            risk_detected=True,
        )


def test_risk_detected_must_be_boolean():
    with pytest.raises(
        ValueError,
        match="boolean",
    ):
        create_leakage_finding(
            category="proxy",
            title="Proxy Check",
            message="Test message.",
            severity="high",
            risk_detected="yes",
        )


def test_detected_risk_cannot_have_none_severity():
    with pytest.raises(
        ValueError,
        match="cannot have severity 'none'",
    ):
        create_leakage_finding(
            category="proxy",
            title="Proxy Check",
            message="Test message.",
            severity="none",
            risk_detected=True,
        )


def test_safe_finding_requires_none_severity():
    with pytest.raises(
        ValueError,
        match="must have severity 'none'",
    ):
        create_leakage_finding(
            category="proxy",
            title="Proxy Check",
            message="Test message.",
            severity="medium",
            risk_detected=False,
        )


def test_invalid_recommendation_rejected():
    with pytest.raises(
        ValueError,
        match="Recommendation must be",
    ):
        create_leakage_finding(
            category="group",
            title="Group Check",
            message="Test message.",
            severity="high",
            risk_detected=True,
            recommendation=123,
        )


def test_invalid_details_rejected():
    with pytest.raises(
        ValueError,
        match="details must be a dictionary",
    ):
        create_leakage_finding(
            category="group",
            title="Group Check",
            message="Test message.",
            severity="high",
            risk_detected=True,
            details=["group_1"],
        )


def test_empty_findings_are_safe():
    result = assess_overall_risk([])

    assert result["overall_risk"] == "none"
    assert result["risk_detected"] is False
    assert result["finding_count"] == 0
    assert result["risky_finding_count"] == 0


def test_safe_findings_produce_no_risk():
    finding = create_leakage_finding(
        category="temporal",
        title="Temporal Check",
        message="No risk detected.",
        severity="none",
        risk_detected=False,
    )

    result = assess_overall_risk([finding])

    assert result["overall_risk"] == "none"
    assert result["risk_detected"] is False
    assert result["risky_finding_count"] == 0


def test_low_risk_assessment():
    finding = create_leakage_finding(
        category="proxy",
        title="Proxy Check",
        message="Low risk detected.",
        severity="low",
        risk_detected=True,
    )

    result = assess_overall_risk([finding])

    assert result["overall_risk"] == "low"
    assert result["severity_counts"]["low"] == 1


def test_medium_risk_assessment():
    finding = create_leakage_finding(
        category="proxy",
        title="Proxy Check",
        message="Medium risk detected.",
        severity="medium",
        risk_detected=True,
    )

    result = assess_overall_risk([finding])

    assert result["overall_risk"] == "medium"
    assert result["severity_counts"]["medium"] == 1


def test_high_risk_assessment():
    finding = create_leakage_finding(
        category="group",
        title="Group Check",
        message="High risk detected.",
        severity="high",
        risk_detected=True,
    )

    result = assess_overall_risk([finding])

    assert result["overall_risk"] == "high"
    assert result["severity_counts"]["high"] == 1


def test_highest_severity_determines_overall_risk():
    findings = [
        create_leakage_finding(
            category="proxy",
            title="Proxy Check",
            message="Medium risk.",
            severity="medium",
            risk_detected=True,
        ),
        create_leakage_finding(
            category="group",
            title="Group Check",
            message="High risk.",
            severity="high",
            risk_detected=True,
        ),
        create_leakage_finding(
            category="temporal",
            title="Temporal Check",
            message="No risk.",
            severity="none",
            risk_detected=False,
        ),
    ]

    result = assess_overall_risk(findings)

    assert result["overall_risk"] == "high"
    assert result["finding_count"] == 3
    assert result["risky_finding_count"] == 2
    assert result["severity_counts"] == {
        "low": 0,
        "medium": 1,
        "high": 1,
    }


def test_findings_must_be_list():
    with pytest.raises(
        ValueError,
        match="provided as a list",
    ):
        assess_overall_risk({})


def test_each_finding_must_be_dictionary():
    with pytest.raises(
        ValueError,
        match="must be a dictionary",
    ):
        assess_overall_risk(["invalid"])


def test_missing_required_finding_field():
    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        assess_overall_risk([
            {
                "category": "group",
                "title": "Group Check",
            }
        ])


def test_assessment_rejects_invalid_severity():
    finding = {
        "category": "proxy",
        "title": "Proxy Check",
        "message": "Test message.",
        "severity": "critical",
        "risk_detected": True,
    }

    with pytest.raises(
        ValueError,
        match="unsupported severity",
    ):
        assess_overall_risk([finding])


def test_assessment_rejects_invalid_risk_flag():
    finding = {
        "category": "proxy",
        "title": "Proxy Check",
        "message": "Test message.",
        "severity": "high",
        "risk_detected": 1,
    }

    with pytest.raises(
        ValueError,
        match="must be boolean",
    ):
        assess_overall_risk([finding])


def test_assessment_metadata():
    finding = create_leakage_finding(
        category="group",
        title="Group Check",
        message="High risk detected.",
        severity="high",
        risk_detected=True,
    )

    result = assess_overall_risk([finding])

    assert result["analysis_type"] == (
        "overall_risk_assessment"
    )
    assert result["findings"] == [finding]
    assert "Overall risk: high" in result["summary"]