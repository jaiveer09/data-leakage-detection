import pandas as pd


def analyze_numerical_target_correlation(
    df,
    target_column,
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Input data must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not target_column:
        raise ValueError("A target column is required.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_column]
    ):
        raise ValueError(
            "Numerical target correlation analysis "
            "requires a numerical target column."
        )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    feature_columns = [
        column
        for column in numerical_columns
        if column != target_column
    ]

    correlations = []

    for column in feature_columns:
        valid_data = df[
            [column, target_column]
        ].dropna()

        if len(valid_data) < 2:
            correlation = None
        elif valid_data[column].nunique() < 2:
            correlation = None
        elif valid_data[target_column].nunique() < 2:
            correlation = None
        else:
            correlation = valid_data[column].corr(
                valid_data[target_column]
            )

            if pd.isna(correlation):
                correlation = None
            else:
                correlation = float(correlation)

        correlations.append({
            "feature": column,
            "correlation": correlation,
            "absolute_correlation": (
                abs(correlation)
                if correlation is not None
                else None
            ),
        })

    correlations.sort(
        key=lambda item: (
            item["absolute_correlation"]
            if item["absolute_correlation"] is not None
            else -1
        ),
        reverse=True,
    )

    return {
        "analysis_type": "numerical_target_correlation",
        "target_column": target_column,
        "numerical_feature_count": len(
            feature_columns
        ),
        "correlations": correlations,
    }

def identify_proxy_candidates(
    correlation_result,
    correlation_threshold=0.9,
):
    if not isinstance(correlation_result, dict):
        raise ValueError(
            "Correlation result must be a dictionary."
        )

    if "correlations" not in correlation_result:
        raise ValueError(
            "Correlation result does not contain correlations."
        )

    if not 0 < correlation_threshold <= 1:
        raise ValueError(
            "Correlation threshold must be greater than 0 "
            "and less than or equal to 1."
        )

    candidates = []

    for item in correlation_result["correlations"]:
        absolute_correlation = item[
            "absolute_correlation"
        ]

        if absolute_correlation is None:
            continue

        if absolute_correlation >= correlation_threshold:
            candidates.append({
                "feature": item["feature"],
                "correlation": item["correlation"],
                "absolute_correlation": (
                    absolute_correlation
                ),
                "reason": (
                    "The feature has a very strong numerical "
                    "relationship with the target and should be "
                    "reviewed for possible proxy or target leakage."
                ),
            })

    return {
        "analysis_type": "proxy_candidate_analysis",
        "correlation_threshold": correlation_threshold,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warning": (
            "Strong correlation alone does not prove data leakage. "
            "Candidate features require additional investigation."
        ),
    }