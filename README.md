# A System for Detecting Data Leakage in Machine Learning Pipelines

A Python-based application for identifying potential data leakage risks in machine learning datasets and evaluation configurations. The system uses Streamlit to provide an interactive interface for dataset analysis, leakage detection, diagnostic modeling, visualizations, and structured reporting.

## Project Overview

Data leakage occurs when information that should not be available during model training or prediction influences the training or evaluation process. This can produce misleading performance results and reduce a model's ability to generalize to unseen data.

The application provides a structured workflow for identifying common leakage risks. It analyzes uploaded datasets and user-selected evaluation settings, generates warnings, and provides supporting evidence to help users investigate potential problems.

The system focuses on temporal leakage, group leakage, numerical proxy leakage, preprocessing configuration risks, and cross-validation configuration risks. It also allows users to compare diagnostic model performance with and without suspected leakage features.

## Features

- CSV dataset upload and validation
- Dataset profiling, including missing values, data types, duplicate rows, and column statistics
- Random, time-based, and group-based train-test splitting
- Temporal leakage detection using timestamp information
- Group leakage detection using overlapping entity identifiers
- Numerical proxy leakage analysis using feature-to-target correlations
- Preprocessing configuration validation
- Cross-validation configuration validation
- Diagnostic modeling using Logistic Regression or Linear Regression
- Coefficient-based feature importance analysis
- Clean-versus-leaked model performance comparison
- Correlation heatmaps and feature distribution visualizations
- Categorized leakage warnings, recommendations, and overall risk assessment
- Downloadable structured JSON analysis reports
- Validation messages for invalid datasets and evaluation configurations

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Matplotlib and Seaborn
- pytest

## Installation

### Prerequisites

Python and pip must be installed on the computer. The application is designed to run locally and does not require specialized hardware.

### 1. Clone the Repository

```bash
git clone https://github.com/jaiveer09/data-leakage-detection.git
cd data-leakage-detection
```

### 2. Create a Virtual Environment

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Application

From the project root directory, run:

```bash
python -m streamlit run app/app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`, which can be opened in a web browser.

## How to Use

1. Upload a CSV dataset using the file uploader.
2. Select the target or label column.
3. Optionally select a timestamp column and a group/entity column.
4. Choose a train-test split strategy and test-set proportion.
5. Select the cross-validation strategy and number of folds.
6. Configure preprocessing settings, if applicable.
7. Optionally select numerical features suspected of containing leakage.
8. Click **Analyze Dataset**.
9. Review the analysis results using the dashboard tabs.
10. Download the structured analysis report in JSON format if needed.

### Analysis Dashboard

After an analysis is completed, the application organizes its results into 11 tabs.

| Tab | Description |
|---|---|
| Overview | Displays leakage findings, overall risk, severity levels, and recommendations. |
| Diagnostic Modeling | Displays the selected diagnostic model, evaluation score, and feature importance results. |
| Model Comparison | Compares diagnostic model performance with and without suspected leakage features. |
| Dataset Profile | Displays dataset dimensions, data types, missing values, and other profiling information. |
| Pipeline Validation | Displays evaluation settings, preprocessing risk assessments, and cross-validation configuration checks. |
| Split Validation | Displays split validation results and the actual training and testing datasets. |
| Temporal Leakage | Displays temporal leakage findings when a timestamp column is selected. |
| Group Leakage | Displays group overlap findings when a group/entity column is selected. |
| Proxy Leakage | Displays numerical correlations, possible proxy leakage candidates, and a correlation heatmap. |
| Statistics & Distributions | Displays numerical and categorical statistics and allows users to visualize individual feature distributions. |
| JSON Report | Provides a downloadable structured analysis report. |

The Temporal Leakage and Group Leakage tabs display informational messages when their corresponding optional columns are not selected. The Model Comparison tab provides guidance when no suspected leakage features are selected.

### Model Performance Comparison

The application allows users to select one or more numerical features suspected of introducing leakage.

When these features are selected, the system evaluates two diagnostic models using the same training and testing datasets. One model excludes the selected features, while the other includes them.

The application displays both evaluation scores, the score difference, and a bar chart.

A higher score after including suspected features may indicate a leakage problem, but the performance difference alone does not establish that leakage exists. The selected features must also be reviewed to determine whether they would legitimately be available at prediction time.

### Structured Analysis Report

The application allows users to download a JSON report containing:

- Dataset information and profiling results
- Evaluation configuration
- Preprocessing settings
- Overall leakage risk assessment
- Individual leakage findings and severity levels
- Warnings and recommendations
- Diagnostic model results, when available
- Model comparison results, when requested and available
- Explanatory limitations

The JSON report allows users to review the analysis results outside the Streamlit application.

## Running Automated Tests

The project uses pytest for automated testing.

From the project root directory, run:

```bash
python -m pytest -q
```

To check the Streamlit application for Python syntax errors, run:

```bash
python -m py_compile app/app.py
```

As of October 28, 2026, the complete automated test suite contained **230 tests**, all of which passed.

Additional end-to-end testing was performed using the Streamlit interface. The testing covered dataset loading, configuration validation, leakage detection, diagnostic modeling, model comparison, visualizations, JSON report downloading, error handling, and repeated analyses.

An initial acceptance-testing exercise covered 36 scenarios and identified a discrepancy between estimated and actual train-test row counts for certain random splits. The issue was corrected, and additional regression tests were added.

Following the dashboard redesign, further regression testing covered the original scenarios and the new tabbed interface. Two minor informational-message issues were identified and subsequently corrected.

The final interface changes were also manually checked before being merged into the main branch.

## Limitations

The application is intended to identify potential leakage risks rather than guarantee that all forms of data leakage will be detected.

Proxy leakage analysis uses correlations between numerical features and numerical target variables. Strong correlations do not automatically establish leakage, and some target-derived features may not exceed the selected correlation threshold.

Temporal leakage detection depends on the timestamp information provided by the user. The level of timestamp detail can affect the reported results.

Preprocessing and cross-validation checks evaluate user-selected configuration settings rather than inspecting or automatically correcting arbitrary external machine learning pipelines. The cross-validation validator assesses whether the selected strategy is appropriate, but it does not execute a complete cross-validation experiment.

Diagnostic modeling focuses on numerical input features and uses lightweight linear models. The application is intended for structured tabular datasets of manageable size and is not designed as a large-scale production monitoring system.

The overall leakage risk assessment summarizes the risks identified by the implemented checks. A low-risk result or the absence of reported findings does not guarantee that a dataset or machine learning pipeline is free from leakage.

## Project Status

The core application implementation has been completed.

The system includes dataset validation, profiling, train-test splitting, leakage detection, pipeline configuration validation, diagnostic modeling, feature importance analysis, model performance comparison, visualizations, risk assessment, and structured JSON report generation.

The Streamlit interface has been organized into an 11-tab dashboard. Automated testing and end-to-end interface testing have been completed, and the identified functional and informational-message issues have been corrected.

The completed application code has been committed and merged into the main GitHub branch.

The remaining project work focuses on preparing the final academic report and presentation materials.

## Author

Jaiveer Kapadia