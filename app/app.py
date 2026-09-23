import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from modules.data_loader import load_csv, validate_columns
from modules.config import create_config
from modules.profiler import profile_dataset, get_column_statistics
from modules.split_validator import (
    validate_split_configuration,
    create_train_test_split,
)
from modules.temporal_detector import detect_temporal_leakage


st.set_page_config(
    page_title="Data Leakage Detection System",
    layout="wide",
)

st.title("Data Leakage Detection System")
st.write(
    "Upload a CSV dataset and configure the columns that will be used "
    "for leakage analysis."
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)

        st.success("Dataset loaded successfully.")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        st.subheader("Column Configuration")

        target_column = st.selectbox(
            "Target / Label Column",
            options=columns,
        )

        optional_columns = ["None"] + columns

        timestamp_selection = st.selectbox(
            "Timestamp Column (Optional, used for Temporal Leakage Analysis)",
            options=optional_columns,
        )

        group_selection = st.selectbox(
            "Group / Entity Column (Required for Group-Based Split)",
            options=optional_columns,
        )

        timestamp_column = (
            None
            if timestamp_selection == "None"
            else timestamp_selection
        )

        group_column = (
            None
            if group_selection == "None"
            else group_selection
        )
        
        st.subheader("Evaluation Configuration")

        split_type = st.selectbox(
            "Split Type",
            options=["random", "time", "group"],
        )

        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
        )

        cv_folds = st.number_input(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
        )

        if st.button("Analyze Dataset"):
            try:
                validate_columns(
                    df,
                    target_column,
                    timestamp_column,
                    group_column,
                )
                
                config = create_config(
                    target_column=target_column,
                    timestamp_column=timestamp_column,
                    group_column=group_column,
                    split_type=split_type,
                    test_size=test_size,
                    cv_folds=int(cv_folds),
                )

                split_validation = validate_split_configuration(
                    df,
                    config,
                )
                
                train_df, test_df = create_train_test_split(df, config)
                
                temporal_result = None

                if config["timestamp_column"] is not None:
                    temporal_result = detect_temporal_leakage(
                        train_df,
                        test_df,
                        config["timestamp_column"],
                        config["split_type"],
                    )
                    
                statistics = get_column_statistics(df)

                profile = profile_dataset(
                    df,
                    target_column,
                )

                st.subheader("Dataset Profile")

                col1, col2, col3 = st.columns(3)

                col1.metric("Rows", profile["rows"])
                col2.metric("Columns", profile["columns"])
                col3.metric(
                    "Duplicate Rows",
                    profile["duplicate_rows"],
                )

                st.write("**Column Data Types**")
                st.json(profile["data_types"])

                st.write("**Missing Values**")
                st.json(profile["missing_values"])

                st.write(
                    "**Numerical Columns:**",
                    profile["numerical_columns"],
                )

                st.write(
                    "**Categorical Columns:**",
                    profile["categorical_columns"],
                )

                st.write(
                    "**Target Unique Values:**",
                    profile["target_unique_values"],
                )

                st.write(
                    "**Target Missing Values:**",
                    profile["target_missing_values"],
                )
                
                st.subheader("Evaluation Settings")

                st.write("**Split Type:**", config["split_type"])
                st.write("**Test Size:**", config["test_size"])
                st.write("**Cross-Validation Folds:**", config["cv_folds"])
                
                st.subheader("Split Validation")

                st.success("Split configuration is valid.")

                if split_validation["split_type"] == "random":
                    st.write(
                        "**Estimated Training Rows:**",
                        split_validation["train_rows"],
                    )
                    st.write(
                        "**Estimated Testing Rows:**",
                        split_validation["test_rows"],
                    )

                elif split_validation["split_type"] == "time":
                    st.write(
                        "**Timestamp Column:**",
                        split_validation["timestamp_column"],
                    )
                    st.write(
                        "**Estimated Training Rows:**",
                        split_validation["train_rows"],
                    )
                    st.write(
                        "**Estimated Testing Rows:**",
                        split_validation["test_rows"],
                    )
                    st.write(
                        "**Earliest Timestamp:**",
                        split_validation["earliest_timestamp"],
                    )
                    st.write(
                        "**Latest Timestamp:**",
                        split_validation["latest_timestamp"],
                    )

                elif split_validation["split_type"] == "group":
                    st.write(
                        "**Group Column:**",
                        split_validation["group_column"],
                    )
                    st.write(
                        "**Unique Groups:**",
                        split_validation["unique_groups"],
                    )

                st.subheader("Train / Test Split")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Training Set**")
                    st.write(f"Rows: {len(train_df)}")
                    st.dataframe(train_df, use_container_width=True)

                with col2:
                    st.write("**Testing Set**")
                    st.write(f"Rows: {len(test_df)}")
                    st.dataframe(test_df, use_container_width=True)
                    
                if temporal_result is not None:
                    st.subheader("Temporal Leakage Analysis")

                    st.write("**Split Strategy Assessment**")

                    if temporal_result["configuration_risk"]:
                        st.warning(
                            temporal_result["configuration_message"]
                        )
                    else:
                        st.success(
                            temporal_result["configuration_message"]
                        )

                    st.write("**Observed Train / Test Analysis**")

                    if temporal_result["temporal_overlap"]:
                        st.error(temporal_result["message"])
                    else:
                        st.success(temporal_result["message"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(
                            "**Earliest Training Timestamp:**",
                            temporal_result[
                                "earliest_train_timestamp"
                            ],
                        )

                        st.write(
                            "**Latest Training Timestamp:**",
                            temporal_result[
                                "latest_train_timestamp"
                            ],
                        )

                    with col2:
                        st.write(
                            "**Earliest Testing Timestamp:**",
                            temporal_result[
                                "earliest_test_timestamp"
                            ],
                        )

                        st.write(
                            "**Latest Testing Timestamp:**",
                            temporal_result[
                                "latest_test_timestamp"
                            ],
                        )

                    st.write(
                        "**Observed Temporal Overlap:**",
                        "Yes"
                        if temporal_result["temporal_overlap"]
                        else "No",
                    )

                    st.write(
                        "**Severity:**",
                        temporal_result["severity"].capitalize(),
                    )

                    if temporal_result["recommendation"]:
                        st.info(
                            "Recommendation: "
                            + temporal_result["recommendation"]
                        )
                    
                st.subheader("Column Statistics")

                st.write("**Numerical Statistics**")
                st.json(statistics["numerical"])

                st.write("**Categorical Statistics**")
                st.json(statistics["categorical"])

            except ValueError as e:
                st.error(str(e))

    except ValueError as e:
        st.error(str(e))