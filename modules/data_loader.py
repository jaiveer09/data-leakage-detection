
import csv
import io
import os

import pandas as pd


def load_csv(file):
    try:
        if isinstance(file, (str, os.PathLike)):
            with open(file, "rb") as csv_file:
                raw_data = csv_file.read()
        else:
            file.seek(0)
            raw_data = file.read()

        if isinstance(raw_data, bytes):
            csv_text = raw_data.decode("utf-8-sig")
        else:
            csv_text = raw_data.lstrip("\ufeff")

        rows = csv.reader(
            io.StringIO(csv_text, newline=""),
            strict=True,
        )

        header = None

        for row in rows:
            if not row:
                continue

            if header is None:
                header = row

                if len(set(header)) != len(header):
                    raise ValueError(
                        "CSV contains duplicate column headers."
                    )

                if any(
                    not column.strip() for column in header
                ):
                    raise ValueError(
                        "CSV contains a missing or blank column header."
                    )

                continue

            if len(row) != len(header):
                raise ValueError(
                    "CSV contains a malformed row with an "
                    "incorrect number of fields."
                )

        if header is None:
            raise ValueError(
                "CSV file is empty; no columns or data were found."
            )

        df = pd.read_csv(io.StringIO(csv_text))

        if df.empty:
            raise ValueError(
                "CSV file is empty; expected at least one row of data."
            )

        if df.isna().all(axis=None):
            raise ValueError(
                "CSV contains no non-missing data values."
            )

        return df

    except pd.errors.EmptyDataError as e:
        raise ValueError(
            "CSV file is empty; no columns or data were found."
        ) from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to load CSV from {file!r}."
        ) from e


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
