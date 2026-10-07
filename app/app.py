import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader import load_csv, validate_columns
from modules.config import create_config
from modules.profiler import profile_dataset, get_column_statistics
from modules.split_validator import (
    validate_split_configuration,
    create_train_test_split,
)
from modules.temporal_detector import detect_temporal_leakage
from modules.group_detector import detect_group_leakage
from modules.proxy_detector import (
    analyze_numerical_target_correlation,
    identify_proxy_candidates,
)

from modules.pipeline_validator import (
    validate_preprocessing_configuration,
    validate_cross_validation_configuration,
)

from modules.risk_assessor import (
    create_leakage_finding,
    assess_overall_risk,
)

from modules.diagnostic_model import (
    run_diagnostic_model,
    analyze_feature_importance,
)

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
        
        cv_strategy = st.selectbox(
            "Cross-Validation Strategy",
            options=[
                "kfold",
                "time_series",
                "group",
            ],
        )
        
        st.subheader("Preprocessing Configuration")

        use_scaling = st.checkbox(
            "Use feature scaling"
        )

        scaling_scope = None

        if use_scaling:
            scaling_scope = st.selectbox(
                "Feature Scaling Fit Scope",
                options=[
                    "training",
                    "full_dataset",
                ],
            )

        use_imputation = st.checkbox(
            "Use missing-value imputation"
        )

        imputation_scope = None

        if use_imputation:
            imputation_scope = st.selectbox(
                "Imputation Fit Scope",
                options=[
                    "training",
                    "full_dataset",
                ],
            )

        if st.button("Analyze Dataset"):
            try:
                
                preprocessing_steps = []

                if use_scaling:
                    preprocessing_steps.append({
                        "name": "Feature Scaling",
                        "fit_scope": scaling_scope,
                    })

                if use_imputation:
                    preprocessing_steps.append({
                        "name": "Missing-Value Imputation",
                        "fit_scope": imputation_scope,
                    })
                    
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
                
                preprocessing_result = (
                    validate_preprocessing_configuration(
                        preprocessing_steps
                    )
                )

                cross_validation_result = (
                    validate_cross_validation_configuration(
                        split_type=config["split_type"],
                        cv_strategy=cv_strategy,
                        cv_folds=config["cv_folds"],
                    )
                )

                split_validation = validate_split_configuration(
                    df,
                    config,
                )
                
                train_df, test_df = create_train_test_split(df, config)
                
                group_result = None

                if config["group_column"] is not None:
                    group_result = detect_group_leakage(
                        train_df,
                        test_df,
                        config["group_column"],
                    )
                
                temporal_result = None

                if config["timestamp_column"] is not None:
                    temporal_result = detect_temporal_leakage(
                        train_df,
                        test_df,
                        config["timestamp_column"],
                        config["split_type"],
                    )
                
                proxy_correlation_result = None
                proxy_candidate_result = None
                leakage_findings = []

                if pd.api.types.is_numeric_dtype(df[target_column]):
                    proxy_correlation_result = (
                        analyze_numerical_target_correlation(
                            df,
                            target_column,
                        )
                    )

                    proxy_candidate_result = identify_proxy_candidates(
                        proxy_correlation_result
                    )
                

                if temporal_result is not None:
                    leakage_findings.append(
                        create_leakage_finding(
                            category="Temporal Leakage",
                            title="Temporal Split Analysis",
                            message=temporal_result["message"],
                            severity=temporal_result["severity"],
                            risk_detected=temporal_result[
                                "risk_detected"
                            ],
                            recommendation=temporal_result[
                                "recommendation"
                            ],
                            details={
                                "split_type": temporal_result[
                                    "split_type"
                                ],
                                "configuration_risk": temporal_result[
                                    "configuration_risk"
                                ],
                            },
                        )
                    )

                if group_result is not None:
                    leakage_findings.append(
                        create_leakage_finding(
                            category="Group Leakage",
                            title="Group Separation Analysis",
                            message=group_result["message"],
                            severity=group_result["severity"],
                            risk_detected=group_result[
                                "risk_detected"
                            ],
                            recommendation=group_result[
                                "recommendation"
                            ],
                            details={
                                "group_column": group_result[
                                    "group_column"
                                ],
                                "overlap_count": group_result[
                                    "overlap_count"
                                ],
                            },
                        )
                    )

                if proxy_candidate_result is not None:
                    proxy_risk_detected = (
                        proxy_candidate_result[
                            "candidate_count"
                        ] > 0
                    )

                    leakage_findings.append(
                        create_leakage_finding(
                            category="Proxy Leakage",
                            title="Proxy Feature Analysis",
                            message=(
                                f"{proxy_candidate_result['candidate_count']} "
                                "possible proxy leakage candidate(s) "
                                "were identified."
                                if proxy_risk_detected
                                else
                                "No proxy leakage candidates were "
                                "identified using the current "
                                "correlation threshold."
                            ),
                            severity=(
                                "medium"
                                if proxy_risk_detected
                                else "none"
                            ),
                            risk_detected=proxy_risk_detected,
                            recommendation=(
                                "Review strongly correlated features "
                                "to determine whether they encode "
                                "target information that would not be "
                                "available at prediction time."
                                if proxy_risk_detected
                                else None
                            ),
                            details={
                                "candidate_count":
                                    proxy_candidate_result[
                                        "candidate_count"
                                    ],
                                "correlation_threshold":
                                    proxy_candidate_result[
                                        "correlation_threshold"
                                    ],
                            },
                        )
                    )

                leakage_findings.append(
                    create_leakage_finding(
                        category="Preprocessing Leakage",
                        title="Preprocessing Configuration",
                        message=preprocessing_result["summary"],
                        severity=preprocessing_result["severity"],
                        risk_detected=preprocessing_result[
                            "risk_detected"
                        ],
                        recommendation=(
                            "Fit preprocessing operations using "
                            "training data only."
                            if preprocessing_result[
                                "risk_detected"
                            ]
                            else None
                        ),
                        details={
                            "step_count": preprocessing_result[
                                "step_count"
                            ],
                            "risky_step_count":
                                preprocessing_result[
                                    "risky_step_count"
                                ],
                        },
                    )
                )

                leakage_findings.append(
                    create_leakage_finding(
                        category="Cross-Validation Risk",
                        title="Cross-Validation Configuration",
                        message=cross_validation_result["message"],
                        severity=cross_validation_result["severity"],
                        risk_detected=cross_validation_result[
                            "risk_detected"
                        ],
                        recommendation=cross_validation_result[
                            "recommendation"
                        ],
                        details={
                            "selected_strategy":
                                cross_validation_result[
                                    "cv_strategy"
                                ],
                            "recommended_strategy":
                                cross_validation_result[
                                    "recommended_strategy"
                                ],
                            "cv_folds":
                                cross_validation_result[
                                    "cv_folds"
                                ],
                        },
                    )
                )
                    
                overall_risk_result = assess_overall_risk(
                    leakage_findings
                )
                    
                diagnostic_result = None
                feature_importance_result = None
                diagnostic_error = None

                diagnostic_excluded_columns = []

                if config["timestamp_column"] is not None:
                    diagnostic_excluded_columns.append(
                        config["timestamp_column"]
                    )

                if config["group_column"] is not None:
                    diagnostic_excluded_columns.append(
                        config["group_column"]
                    )

                try:
                    diagnostic_result = run_diagnostic_model(
                        train_df,
                        test_df,
                        target_column=target_column,
                        excluded_columns=diagnostic_excluded_columns,
                    )

                    feature_importance_result = (
                        analyze_feature_importance(
                            diagnostic_result
                        )
                    )

                except ValueError as diagnostic_exception:
                    diagnostic_error = str(
                        diagnostic_exception
                    )
                        
                statistics = get_column_statistics(df)

                profile = profile_dataset(
                    df,
                    target_column,
                )
                
                st.subheader("Leakage Findings Summary")

                risky_finding_count = sum(
                    finding["risk_detected"]
                    for finding in leakage_findings
                )

                summary_col1, summary_col2 = st.columns(2)

                summary_col1.metric(
                    "Checks Performed",
                    len(leakage_findings),
                )

                summary_col2.metric(
                    "Risks Identified",
                    risky_finding_count,
                )
                
                st.write("**Overall Leakage Risk**")

                overall_risk = overall_risk_result[
                    "overall_risk"
                ]

                if overall_risk == "high":
                    st.error(
                        overall_risk_result["summary"]
                    )
                elif overall_risk == "medium":
                    st.warning(
                        overall_risk_result["summary"]
                    )
                elif overall_risk == "low":
                    st.warning(
                        overall_risk_result["summary"]
                    )
                else:
                    st.success(
                        overall_risk_result["summary"]
                    )

                severity_counts = overall_risk_result[
                    "severity_counts"
                ]

                risk_col1, risk_col2, risk_col3 = st.columns(3)

                risk_col1.metric(
                    "High Risk Findings",
                    severity_counts["high"],
                )

                risk_col2.metric(
                    "Medium Risk Findings",
                    severity_counts["medium"],
                )

                risk_col3.metric(
                    "Low Risk Findings",
                    severity_counts["low"],
                )

                for finding in leakage_findings:
                    st.write(
                        f"**{finding['category']} — "
                        f"{finding['title']}**"
                    )

                    st.write(
                        "**Severity:**",
                        finding["severity"].capitalize(),
                    )

                    if finding["risk_detected"]:
                        st.warning(finding["message"])
                    else:
                        st.success(finding["message"])

                    if finding["recommendation"]:
                        st.info(
                            "Recommendation: "
                            + finding["recommendation"]
                        )

                    with st.expander("View evidence"):
                        if finding["details"]:
                            st.json(finding["details"])
                        else:
                            st.write(
                                "No additional evidence details "
                                "are available."
                            )

                st.subheader("Diagnostic Model Analysis")

                if diagnostic_result is None:
                    st.info(
                        "Diagnostic modeling could not be "
                        "performed for this dataset."
                    )

                    if diagnostic_error:
                        st.write(
                            "**Reason:**",
                            diagnostic_error,
                        )

                else:
                    diagnostic_col1, diagnostic_col2, diagnostic_col3 = (
                        st.columns(3)
                    )

                    diagnostic_col1.metric(
                        "Task Type",
                        diagnostic_result[
                            "task_type"
                        ].capitalize(),
                    )

                    diagnostic_col2.metric(
                        "Model",
                        diagnostic_result[
                            "model_name"
                        ],
                    )

                    diagnostic_col3.metric(
                        diagnostic_result[
                            "metric_name"
                        ].upper(),
                        f"{diagnostic_result['score']:.4f}",
                    )

                    st.write(
                        "**Features Used:**",
                        diagnostic_result[
                            "feature_columns"
                        ],
                    )

                    st.write(
                        "**Training Rows:**",
                        diagnostic_result["train_rows"],
                    )

                    st.write(
                        "**Testing Rows:**",
                        diagnostic_result["test_rows"],
                    )

                    if feature_importance_result is not None:
                        st.write(
                            "**Diagnostic Feature Importance**"
                        )

                        importance_df = pd.DataFrame(
                            feature_importance_result[
                                "feature_importance"
                            ]
                        )

                        if not importance_df.empty:
                            importance_df = importance_df.rename(
                                columns={
                                    "feature": "Feature",
                                    "importance": "Importance",
                                }
                            )

                            st.dataframe(
                                importance_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                            chart_data = (
                                importance_df
                                .set_index("Feature")
                                ["Importance"]
                            )

                            st.bar_chart(chart_data)

                            st.info(
                                "Feature importance values are based "
                                "on the absolute model coefficients. "
                                "Higher importance indicates greater "
                                "influence on the diagnostic model, "
                                "but does not by itself prove leakage."
                            )
                        else:
                            st.info(
                                "No feature importance values "
                                "are available."
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
                
                st.subheader("Pipeline Validation")

                st.write(
                    "**Preprocessing Leakage Assessment**"
                )

                if preprocessing_result["risk_detected"]:
                    st.warning(
                        preprocessing_result["summary"]
                    )
                else:
                    st.success(
                        preprocessing_result["summary"]
                    )

                for finding in preprocessing_result["findings"]:
                    st.write(
                        f"**{finding['step']}**"
                    )

                    if finding["risk_detected"]:
                        st.warning(finding["message"])
                    else:
                        st.success(finding["message"])

                    if finding["recommendation"]:
                        st.info(
                            "Recommendation: "
                            + finding["recommendation"]
                        )

                st.write(
                    "**Cross-Validation Assessment**"
                )

                if cross_validation_result["risk_detected"]:
                    st.warning(
                        cross_validation_result["message"]
                    )
                else:
                    st.success(
                        cross_validation_result["message"]
                    )

                st.write(
                    "**Selected CV Strategy:**",
                    cross_validation_result["cv_strategy"],
                )

                st.write(
                    "**Recommended CV Strategy:**",
                    cross_validation_result[
                        "recommended_strategy"
                    ],
                )

                st.write(
                    "**CV Folds:**",
                    cross_validation_result["cv_folds"],
                )

                if cross_validation_result["recommendation"]:
                    st.info(
                        "Recommendation: "
                        + cross_validation_result[
                            "recommendation"
                        ]
                    )
                    
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
                
                if group_result is not None:
                    st.subheader("Group Leakage Analysis")

                    if group_result["leakage_detected"]:
                        st.error(group_result["message"])
                    else:
                        st.success(group_result["message"])

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Training Groups",
                        group_result["train_group_count"],
                    )

                    col2.metric(
                        "Testing Groups",
                        group_result["test_group_count"],
                    )

                    col3.metric(
                        "Overlapping Groups",
                        group_result["overlap_count"],
                    )

                    st.write(
                        "**Group Column:**",
                        group_result["group_column"],
                    )

                    st.write(
                        "**Observed Group Leakage:**",
                        "Yes"
                        if group_result["leakage_detected"]
                        else "No",
                    )

                    st.write(
                        "**Severity:**",
                        group_result["severity"].capitalize(),
                    )

                    if group_result["overlapping_groups"]:
                        st.write(
                            "**Overlapping Group Values:**",
                            group_result["overlapping_groups"],
                        )

                    if group_result["recommendation"]:
                        st.info(
                            "Recommendation: "
                            + group_result["recommendation"]
                        )
                st.subheader("Proxy Leakage Analysis")

                if proxy_correlation_result is None:
                    st.info(
                        "Proxy leakage correlation analysis was not performed "
                        "because the selected target column is not numerical."
                    )

                else:
                    st.write(
                        "**Numerical Feature-to-Target Correlations**"
                    )

                    correlation_rows = []

                    for item in proxy_correlation_result["correlations"]:
                        correlation_rows.append({
                            "Feature": item["feature"],
                            "Correlation": item["correlation"],
                            "Absolute Correlation": (
                                item["absolute_correlation"]
                            ),
                        })
                    if correlation_rows:
                        correlation_df = pd.DataFrame(
                            correlation_rows
                        )

                        st.dataframe(
                            correlation_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        valid_correlation_rows = [
                            item
                            for item in proxy_correlation_result[
                                "correlations"
                            ]
                            if item["correlation"] is not None
                        ]

                        if valid_correlation_rows:
                            st.write(
                                "**Feature-to-Target Correlation Heatmap**"
                            )

                            heatmap_data = pd.DataFrame(
                                {
                                    item["feature"]: [
                                        item["correlation"]
                                    ]
                                    for item in valid_correlation_rows
                                },
                                index=[target_column],
                            )

                            figure, axis = plt.subplots(
                                figsize=(
                                    max(
                                        6,
                                        len(valid_correlation_rows) * 1.2,
                                    ),
                                    1.8,
                                )
                            )

                            sns.heatmap(
                                heatmap_data,
                                annot=True,
                                fmt=".2f",
                                cmap="coolwarm",
                                center=0,
                                vmin=-1,
                                vmax=1,
                                ax=axis,
                            )

                            axis.set_xlabel("Numerical Features")
                            axis.set_ylabel("Target")
                            axis.set_title(
                                "Numerical Feature-to-Target Correlations"
                            )

                            figure.tight_layout()

                            st.pyplot(figure)

                            plt.close(figure)
                        else:
                            st.info(
                                "No valid correlations are available "
                                "for visualization."
                            )

                    else:
                        st.info(
                            "No numerical feature columns are available "
                            "for correlation analysis."
                        )

                    st.write(
                        "**Proxy Feature Candidate Assessment**"
                    )

                    candidate_count = (
                        proxy_candidate_result["candidate_count"]
                    )

                    if candidate_count > 0:
                        st.warning(
                            f"{candidate_count} possible proxy leakage "
                            "candidate(s) identified."
                        )

                        for candidate in (
                            proxy_candidate_result["candidates"]
                        ):
                            st.write(
                                f"**{candidate['feature']}**"
                            )

                            st.write(
                                "Correlation:",
                                round(
                                    candidate["correlation"],
                                    4,
                                ),
                            )

                            st.write(
                                candidate["reason"]
                            )

                    else:
                        st.success(
                            "No proxy leakage candidates were identified "
                            "using the current correlation threshold."
                        )

                    st.write(
                        "**Correlation Threshold:**",
                        proxy_candidate_result[
                            "correlation_threshold"
                        ],
                    )

                    st.info(
                        proxy_candidate_result["warning"]
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