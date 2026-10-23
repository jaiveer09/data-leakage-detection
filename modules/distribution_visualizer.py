
import matplotlib.pyplot as plt
import pandas as pd


def prepare_distribution(df, column, max_categories=15):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if column not in df.columns:
        raise ValueError(f"Column {column!r} was not found.")

    if not isinstance(max_categories, int) or isinstance(
        max_categories, bool
    ) or max_categories < 1:
        raise ValueError("max_categories must be a positive integer.")

    series = df[column].dropna()

    if series.empty:
        return {
            "type": "empty",
            "column": column,
            "values": None,
            "truncated": False,
        }

    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        values = pd.to_numeric(series, errors="coerce")
        values = values.replace(
            [float("inf"), float("-inf")], float("nan")
        ).dropna()

        if values.empty:
            return {
                "type": "empty",
                "column": column,
                "values": None,
                "truncated": False,
            }

        return {
            "type": "numerical",
            "column": column,
            "values": values,
            "truncated": False,
        }

    category_counts = series.astype(str).value_counts()

    return {
        "type": "categorical",
        "column": column,
        "values": category_counts.head(max_categories),
        "truncated": len(category_counts) > max_categories,
    }


def create_distribution_figure(distribution):
    distribution_type = distribution["type"]
    column = distribution["column"]
    values = distribution["values"]

    if distribution_type == "empty":
        return None

    figure, axis = plt.subplots(figsize=(8, 4))

    if distribution_type == "numerical":
        bins = min(30, max(1, values.nunique()))

        axis.hist(
            values,
            bins=bins,
            edgecolor="black",
        )

        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")

    elif distribution_type == "categorical":
        values.plot.bar(ax=axis)

        axis.set_title(f"Top Categories: {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
        axis.tick_params(axis="x", labelrotation=45)

    else:
        plt.close(figure)
        raise ValueError(
            f"Unsupported distribution type: {distribution_type}"
        )

    figure.tight_layout()
    return figure
