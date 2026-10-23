
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from modules.distribution_visualizer import (
    prepare_distribution,
    create_distribution_figure,
)


def test_numerical_distribution():
    df = pd.DataFrame({"age": [20, 30, 40, 50]})

    result = prepare_distribution(df, "age")

    assert result["type"] == "numerical"
    assert result["values"].tolist() == [20, 30, 40, 50]
    assert result["truncated"] is False


def test_categorical_distribution_counts():
    df = pd.DataFrame({
        "workclass": ["Private", "Government", "Private"]
    })

    result = prepare_distribution(df, "workclass")

    assert result["type"] == "categorical"
    assert result["values"]["Private"] == 2
    assert result["values"]["Government"] == 1


def test_missing_values_excluded():
    df = pd.DataFrame({
        "income": [100, None, 300]
    })

    result = prepare_distribution(df, "income")

    assert result["values"].tolist() == [100, 300]


def test_all_missing_values():
    df = pd.DataFrame({
        "feature": [None, None, None]
    })

    result = prepare_distribution(df, "feature")

    assert result["type"] == "empty"
    assert result["values"] is None


def test_infinite_values_excluded():
    df = pd.DataFrame({
        "feature": [1.0, float("inf"), float("-inf"), 4.0]
    })

    result = prepare_distribution(df, "feature")

    assert result["values"].tolist() == [1.0, 4.0]


def test_category_limit():
    df = pd.DataFrame({
        "category": [f"category_{i}" for i in range(20)]
    })

    result = prepare_distribution(df, "category")

    assert len(result["values"]) == 15
    assert result["truncated"] is True


def test_boolean_column_is_categorical():
    df = pd.DataFrame({
        "flag": [True, False, True]
    })

    result = prepare_distribution(df, "flag")

    assert result["type"] == "categorical"
    assert result["values"]["True"] == 2


def test_missing_column_rejected():
    df = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(ValueError, match="not found"):
        prepare_distribution(df, "income")


def test_invalid_dataframe_rejected():
    with pytest.raises(ValueError, match="DataFrame"):
        prepare_distribution(["age"], "age")


def test_invalid_category_limit_rejected():
    df = pd.DataFrame({"category": ["A", "B"]})

    with pytest.raises(ValueError, match="positive integer"):
        prepare_distribution(df, "category", max_categories=0)


def test_numerical_figure_created():
    df = pd.DataFrame({"age": [20, 30, 40]})
    result = prepare_distribution(df, "age")

    figure = create_distribution_figure(result)

    try:
        assert figure is not None
        assert figure.axes[0].get_title() == "Distribution of age"
        assert len(figure.axes[0].patches) > 0
    finally:
        plt.close(figure)


def test_categorical_figure_created():
    df = pd.DataFrame({
        "workclass": ["Private", "Private", "Government"]
    })
    result = prepare_distribution(df, "workclass")

    figure = create_distribution_figure(result)

    try:
        assert figure is not None
        assert figure.axes[0].get_title() == "Top Categories: workclass"
        assert len(figure.axes[0].patches) == 2
    finally:
        plt.close(figure)


def test_empty_distribution_has_no_figure():
    df = pd.DataFrame({"feature": [None, None]})
    result = prepare_distribution(df, "feature")

    assert create_distribution_figure(result) is None
