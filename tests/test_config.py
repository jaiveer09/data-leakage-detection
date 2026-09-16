import pytest

from modules.config import create_config


def test_valid_random_config():
    config = create_config(
        target_column="target",
        split_type="random",
        test_size=0.2,
        cv_folds=5,
    )

    assert config["target_column"] == "target"
    assert config["split_type"] == "random"
    assert config["test_size"] == 0.2
    assert config["cv_folds"] == 5


def test_valid_time_config():
    config = create_config(
        target_column="target",
        timestamp_column="date",
        split_type="time",
    )

    assert config["split_type"] == "time"
    assert config["timestamp_column"] == "date"


def test_valid_group_config():
    config = create_config(
        target_column="target",
        group_column="customer_id",
        split_type="group",
    )

    assert config["split_type"] == "group"
    assert config["group_column"] == "customer_id"


def test_target_required():
    with pytest.raises(ValueError, match="target column"):
        create_config(target_column=None)


def test_invalid_split_type():
    with pytest.raises(ValueError, match="Invalid split type"):
        create_config(
            target_column="target",
            split_type="invalid",
        )


def test_invalid_test_size_zero():
    with pytest.raises(ValueError, match="test_size"):
        create_config(
            target_column="target",
            test_size=0,
        )


def test_invalid_test_size_one():
    with pytest.raises(ValueError, match="test_size"):
        create_config(
            target_column="target",
            test_size=1,
        )


def test_invalid_cv_folds():
    with pytest.raises(ValueError, match="cv_folds"):
        create_config(
            target_column="target",
            cv_folds=1,
        )


def test_time_split_requires_timestamp():
    with pytest.raises(ValueError, match="timestamp column"):
        create_config(
            target_column="target",
            split_type="time",
        )


def test_group_split_requires_group_column():
    with pytest.raises(ValueError, match="group column"):
        create_config(
            target_column="target",
            split_type="group",
        )