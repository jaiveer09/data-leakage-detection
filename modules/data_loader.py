import pandas as pd


def load_csv(file):
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("CSV file is empty; expected at least one row of data.")
        return df
    except ValueError:
        # Re-raise our empty-file ValueError unchanged
        raise
    except Exception as e:
        raise ValueError(f"Failed to load CSV from {file!r}.") from e


def validate_columns(df, target_column, timestamp_column=None, group_column=None):
    if not target_column:
        raise ValueError("target_column is required.")

    if target_column not in df.columns:
        raise ValueError(
            f"target_column {target_column!r} not found in columns: {list(df.columns)}"
        )

    if timestamp_column is not None and timestamp_column not in df.columns:
        raise ValueError(
            f"timestamp_column {timestamp_column!r} not found in columns: {list(df.columns)}"
        )

    if group_column is not None and group_column not in df.columns:
        raise ValueError(
            f"group_column {group_column!r} not found in columns: {list(df.columns)}"
        )

    return True