import pandas as pd
import pytest

from modules.data_loader import load_csv, validate_columns


def test_load_valid_csv(tmp_path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "age,income,target\n"
        "25,50000,0\n"
        "35,70000,1\n"
    )

    df = load_csv(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["age", "income", "target"]


def test_load_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("age,income,target\n")

    with pytest.raises(ValueError, match="CSV file is empty"):
        load_csv(csv_file)


def test_validate_columns_valid():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "timestamp": ["2026-01-01", "2026-01-02"],
            "customer_id": [101, 102],
            "target": [0, 1],
        }
    )

    result = validate_columns(
        df,
        target_column="target",
        timestamp_column="timestamp",
        group_column="customer_id",
    )

    assert result is True


def test_missing_target_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "income": [50000, 70000],
        }
    )

    with pytest.raises(ValueError, match="target_column"):
        validate_columns(df, target_column="target")


def test_target_column_required():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="target_column is required"):
        validate_columns(df, target_column=None)


def test_invalid_timestamp_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="timestamp_column"):
        validate_columns(
            df,
            target_column="target",
            timestamp_column="timestamp",
        )


def test_invalid_group_column():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="group_column"):
        validate_columns(
            df,
            target_column="target",
            group_column="customer_id",
        )


def test_optional_columns_not_required():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "target": [0, 1],
        }
    )

    result = validate_columns(
        df,
        target_column="target",
    )

    assert result is True
    
    
def test_load_completely_blank_csv(tmp_path):
    csv_file = tmp_path / "blank.csv"
    csv_file.write_text("")

    with pytest.raises(ValueError, match="CSV file is empty"):
        load_csv(csv_file)


def test_load_csv_with_blank_header(tmp_path):
    csv_file = tmp_path / "blank_header.csv"
    csv_file.write_text(
        "age,,target\n"
        "25,50000,0\n"
        "35,70000,1\n"
    )

    with pytest.raises(ValueError, match="blank column header"):
        load_csv(csv_file)


def test_load_csv_with_whitespace_header(tmp_path):
    csv_file = tmp_path / "whitespace_header.csv"
    csv_file.write_text(
        "age,   ,target\n"
        "25,50000,0\n"
    )

    with pytest.raises(ValueError, match="blank column header"):
        load_csv(csv_file)


def test_load_csv_with_all_missing_values(tmp_path):
    csv_file = tmp_path / "all_missing.csv"
    csv_file.write_text(
        "age,income,target\n"
        ",,\n"
        ",,\n"
    )

    with pytest.raises(ValueError, match="no non-missing data"):
        load_csv(csv_file)


def test_load_csv_with_some_missing_values(tmp_path):
    csv_file = tmp_path / "partial_missing.csv"
    csv_file.write_text(
        "age,income,target\n"
        "25,,0\n"
        "35,70000,1\n"
    )

    df = load_csv(csv_file)

    assert len(df) == 2
    assert pd.isna(df.loc[0, "income"])
    assert df.loc[1, "income"] == 70000


def test_duplicate_column_headers_rejected(tmp_path):
    csv_file = tmp_path / "duplicate_headers.csv"
    csv_file.write_text(
        "age,age,target\n"
        "25,50000,0\n"
    )

    with pytest.raises(ValueError, match="duplicate column headers"):
        load_csv(csv_file)


def test_malformed_row_with_extra_field(tmp_path):
    csv_file = tmp_path / "extra_field.csv"
    csv_file.write_text(
        "age,income,target\n"
        "25,50000,0,extra\n"
    )

    with pytest.raises(ValueError, match="malformed row"):
        load_csv(csv_file)


def test_malformed_row_with_missing_field(tmp_path):
    csv_file = tmp_path / "missing_field.csv"
    csv_file.write_text(
        "age,income,target\n"
        "25,50000\n"
    )

    with pytest.raises(ValueError, match="malformed row"):
        load_csv(csv_file)


def test_quoted_commas_are_accepted(tmp_path):
    csv_file = tmp_path / "quoted_commas.csv"
    csv_file.write_text(
        "description,target\n"
        '"New York, USA",1\n'
        '"Phoenix, AZ",0\n'
    )

    df = load_csv(csv_file)

    assert len(df) == 2
    assert df.loc[0, "description"] == "New York, USA"


def test_quoted_newlines_are_accepted(tmp_path):
    csv_file = tmp_path / "quoted_newlines.csv"
    csv_file.write_text(
        "description,target\n"
        '"First line\nSecond line",1\n'
    )

    df = load_csv(csv_file)

    assert len(df) == 1
    assert df.loc[0, "description"].replace("\r\n", "\n") == "First line\nSecond line"


def test_streamlit_style_uploaded_file():
    import io

    uploaded_file = io.BytesIO(
        b"age,income,target\n25,50000,0\n35,70000,1\n"
    )

    df = load_csv(uploaded_file)

    assert len(df) == 2
    assert df.columns.tolist() == ["age", "income", "target"]


def test_utf8_bom_csv_accepted(tmp_path):
    csv_file = tmp_path / "utf8_bom.csv"
    csv_file.write_text(
        "age,income,target\n25,50000,0\n",
        encoding="utf-8-sig",
    )

    df = load_csv(csv_file)

    assert df.columns.tolist() == ["age", "income", "target"]
