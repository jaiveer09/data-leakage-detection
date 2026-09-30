import pytest

from modules.pipeline_validator import (
    validate_preprocessing_configuration,
)


def test_clean_preprocessing_configuration():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "training",
        },
        {
            "name": "SimpleImputer",
            "fit_scope": "training",
        },
    ]

    result = validate_preprocessing_configuration(steps)

    assert result["risk_detected"] is False
    assert result["risky_step_count"] == 0
    assert result["severity"] == "none"
    assert len(result["findings"]) == 2


def test_full_dataset_preprocessing_detects_risk():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "full_dataset",
        }
    ]

    result = validate_preprocessing_configuration(steps)

    assert result["risk_detected"] is True
    assert result["risky_step_count"] == 1
    assert result["severity"] == "high"


def test_mixed_preprocessing_configuration():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "full_dataset",
        },
        {
            "name": "SimpleImputer",
            "fit_scope": "training",
        },
    ]

    result = validate_preprocessing_configuration(steps)

    assert result["risk_detected"] is True
    assert result["risky_step_count"] == 1
    assert len(result["findings"]) == 2

    assert result["findings"][0]["risk_detected"] is True
    assert result["findings"][1]["risk_detected"] is False


def test_multiple_risky_preprocessing_steps():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "full_dataset",
        },
        {
            "name": "SimpleImputer",
            "fit_scope": "full_dataset",
        },
        {
            "name": "OneHotEncoder",
            "fit_scope": "full_dataset",
        },
    ]

    result = validate_preprocessing_configuration(steps)

    assert result["risk_detected"] is True
    assert result["risky_step_count"] == 3
    assert result["severity"] == "high"


def test_empty_preprocessing_steps():
    result = validate_preprocessing_configuration([])

    assert result["step_count"] == 0
    assert result["risky_step_count"] == 0
    assert result["risk_detected"] is False
    assert result["severity"] == "none"
    assert result["findings"] == []


def test_none_preprocessing_steps():
    result = validate_preprocessing_configuration(None)

    assert result["step_count"] == 0
    assert result["risk_detected"] is False


def test_training_step_has_no_recommendation():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "training",
        }
    ]

    result = validate_preprocessing_configuration(steps)

    finding = result["findings"][0]

    assert finding["severity"] == "none"
    assert finding["recommendation"] is None


def test_risky_step_has_recommendation():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "full_dataset",
        }
    ]

    result = validate_preprocessing_configuration(steps)

    finding = result["findings"][0]

    assert finding["severity"] == "high"
    assert finding["recommendation"] is not None


def test_analysis_metadata():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "training",
        }
    ]

    result = validate_preprocessing_configuration(steps)

    assert result["analysis_type"] == (
        "preprocessing_validation"
    )
    assert result["step_count"] == 1


def test_preprocessing_steps_must_be_list():
    with pytest.raises(
        ValueError,
        match="Preprocessing steps must be provided as a list",
    ):
        validate_preprocessing_configuration(
            "StandardScaler"
        )


def test_each_step_must_be_dictionary():
    with pytest.raises(
        ValueError,
        match="Each preprocessing step must be a dictionary",
    ):
        validate_preprocessing_configuration(
            ["StandardScaler"]
        )


def test_preprocessing_step_requires_name():
    steps = [
        {
            "fit_scope": "training",
        }
    ]

    with pytest.raises(
        ValueError,
        match="requires a name",
    ):
        validate_preprocessing_configuration(steps)


def test_preprocessing_step_rejects_empty_name():
    steps = [
        {
            "name": "",
            "fit_scope": "training",
        }
    ]

    with pytest.raises(
        ValueError,
        match="requires a name",
    ):
        validate_preprocessing_configuration(steps)


def test_missing_fit_scope_rejected():
    steps = [
        {
            "name": "StandardScaler",
        }
    ]

    with pytest.raises(
        ValueError,
        match="must specify fit_scope",
    ):
        validate_preprocessing_configuration(steps)


def test_invalid_fit_scope_rejected():
    steps = [
        {
            "name": "StandardScaler",
            "fit_scope": "test",
        }
    ]

    with pytest.raises(
        ValueError,
        match="must specify fit_scope",
    ):
        validate_preprocessing_configuration(steps)