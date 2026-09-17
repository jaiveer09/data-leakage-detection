import pandas as pd


def profile_dataset(df, target_column):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "target_unique_values": int(df[target_column].nunique(dropna=True)),
        "target_missing_values": int(df[target_column].isnull().sum()),
    }

    return profile