import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def determine_task_type(target_series):
    if not isinstance(target_series, pd.Series):
        raise ValueError(
            "Target data must be provided as a pandas Series."
        )

    if target_series.empty:
        raise ValueError(
            "Target data cannot be empty."
        )

    if target_series.isna().any():
        raise ValueError(
            "Target data cannot contain missing values."
        )

    unique_count = target_series.nunique()

    if unique_count < 2:
        raise ValueError(
            "Target data must contain at least two unique values."
        )

    if (
        not pd.api.types.is_numeric_dtype(target_series)
        or unique_count <= 20
    ):
        return "classification"

    return "regression"


def prepare_numerical_features(
    train_df,
    test_df,
    target_column,
    excluded_columns=None,
):
    if excluded_columns is None:
        excluded_columns = []

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found "
            "in the training dataset."
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found "
            "in the testing dataset."
        )

    excluded = set(excluded_columns)
    excluded.add(target_column)

    numerical_columns = [
        column
        for column in train_df.columns
        if column not in excluded
        and column in test_df.columns
        and pd.api.types.is_numeric_dtype(train_df[column])
        and pd.api.types.is_numeric_dtype(test_df[column])
    ]

    if not numerical_columns:
        raise ValueError(
            "No numerical feature columns are available "
            "for diagnostic modeling."
        )

    return numerical_columns


def run_diagnostic_model(
    train_df,
    test_df,
    target_column,
    excluded_columns=None,
):
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError(
            "Training data must be a pandas DataFrame."
        )

    if not isinstance(test_df, pd.DataFrame):
        raise ValueError(
            "Testing data must be a pandas DataFrame."
        )

    if train_df.empty:
        raise ValueError(
            "Training data cannot be empty."
        )

    if test_df.empty:
        raise ValueError(
            "Testing data cannot be empty."
        )

    feature_columns = prepare_numerical_features(
        train_df,
        test_df,
        target_column,
        excluded_columns,
    )

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    task_type = determine_task_type(y_train)

    if y_test.isna().any():
        raise ValueError(
            "Testing target data cannot contain missing values."
        )

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    if task_type == "classification":
        model = LogisticRegression(
            max_iter=1000,
        )
        metric_name = "accuracy"

    else:
        model = LinearRegression()
        metric_name = "r2"

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            model,
        ),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if task_type == "classification":
        score = accuracy_score(
            y_test,
            predictions,
        )
    else:
        score = r2_score(
            y_test,
            predictions,
        )

    return {
        "analysis_type": "diagnostic_model",
        "task_type": task_type,
        "model_name": (
            "Logistic Regression"
            if task_type == "classification"
            else "Linear Regression"
        ),
        "metric_name": metric_name,
        "score": float(score),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "pipeline": pipeline,
    }

def compare_clean_and_leaked_models(
    train_df,
    test_df,
    target_column,
    leaked_feature_columns,
    excluded_columns=None,
):
    if not isinstance(leaked_feature_columns, list):
        raise ValueError(
            "Leaked feature columns must be provided as a list."
        )

    if not leaked_feature_columns:
        raise ValueError(
            "At least one leaked feature column is required."
        )

    missing_columns = [
        column
        for column in leaked_feature_columns
        if column not in train_df.columns
        or column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Leaked feature columns were not found in both "
            "training and testing data: "
            + ", ".join(missing_columns)
        )

    clean_excluded_columns = list(
        excluded_columns or []
    )

    for column in leaked_feature_columns:
        if column not in clean_excluded_columns:
            clean_excluded_columns.append(column)

    clean_result = run_diagnostic_model(
        train_df,
        test_df,
        target_column=target_column,
        excluded_columns=clean_excluded_columns,
    )

    leaked_result = run_diagnostic_model(
        train_df,
        test_df,
        target_column=target_column,
        excluded_columns=excluded_columns,
    )

    if (
        clean_result["task_type"]
        != leaked_result["task_type"]
    ):
        raise ValueError(
            "Clean and leaked diagnostic models must use "
            "the same task type."
        )

    if (
        clean_result["metric_name"]
        != leaked_result["metric_name"]
    ):
        raise ValueError(
            "Clean and leaked diagnostic models must use "
            "the same evaluation metric."
        )

    score_difference = (
        leaked_result["score"]
        - clean_result["score"]
    )

    performance_inflated = score_difference > 0

    return {
        "analysis_type": "clean_leaked_comparison",
        "task_type": clean_result["task_type"],
        "metric_name": clean_result["metric_name"],
        "clean_score": clean_result["score"],
        "leaked_score": leaked_result["score"],
        "score_difference": float(score_difference),
        "performance_inflated": performance_inflated,
        "leaked_feature_columns": leaked_feature_columns,
        "clean_feature_columns": clean_result[
            "feature_columns"
        ],
        "leaked_model_feature_columns": leaked_result[
            "feature_columns"
        ],
        "clean_result": clean_result,
        "leaked_result": leaked_result,
    }

def analyze_feature_importance(
    diagnostic_result,
):
    if not isinstance(diagnostic_result, dict):
        raise ValueError(
            "Diagnostic result must be a dictionary."
        )

    required_fields = {
        "pipeline",
        "feature_columns",
        "task_type",
        "model_name",
    }

    missing_fields = (
        required_fields - diagnostic_result.keys()
    )

    if missing_fields:
        raise ValueError(
            "Diagnostic result is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    pipeline = diagnostic_result["pipeline"]

    if "model" not in pipeline.named_steps:
        raise ValueError(
            "Diagnostic pipeline does not contain a model."
        )

    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError(
            "Diagnostic model does not provide "
            "coefficient-based feature importance."
        )

    feature_columns = diagnostic_result[
        "feature_columns"
    ]

    coefficients = model.coef_

    if coefficients.ndim == 1:
        importance_values = abs(coefficients)
    else:
        importance_values = abs(
            coefficients
        ).mean(axis=0)

    if len(feature_columns) != len(
        importance_values
    ):
        raise ValueError(
            "Feature names and importance values "
            "have different lengths."
        )

    feature_importance = [
        {
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(
            feature_columns,
            importance_values,
        )
    ]

    feature_importance.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {
        "analysis_type": "feature_importance",
        "task_type": diagnostic_result[
            "task_type"
        ],
        "model_name": diagnostic_result[
            "model_name"
        ],
        "feature_count": len(
            feature_importance
        ),
        "feature_importance": feature_importance,
        "top_feature": (
            feature_importance[0]["feature"]
            if feature_importance
            else None
        ),
    }