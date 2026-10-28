# A System for Detecting Data Leakage in Machine Learning Pipelines

A Python-based application for identifying potential data leakage risks in machine learning datasets and evaluation configurations. The system uses Streamlit to provide an interactive interface for dataset analysis, leakage detection, diagnostic modeling, and structured reporting.

## Project Overview

Data leakage occurs when information that should not be available during model training influences the training or evaluation process. This can produce misleading performance results and reduce a model's ability to generalize to unseen data.

The application provides a structured workflow for identifying common leakage risks. It analyzes uploaded datasets and user-selected evaluation settings, generates warnings, and provides supporting evidence to help users investigate potential problems.

## Features

- CSV dataset upload and validation
- Dataset profiling, including missing values, data types, and duplicate rows
- Random, time-based, and group-based train-test splitting
- Temporal leakage detection
- Group leakage detection
- Numerical proxy leakage analysis
- Preprocessing configuration validation
- Cross-validation configuration validation
- Diagnostic modeling using logistic regression or linear regression
- Coefficient-based feature importance analysis
- Clean-versus-leaked model performance comparison
- Correlation heatmaps and feature distribution visualizations
- Categorized leakage warnings and overall risk assessment
- Downloadable JSON analysis reports

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

On Windows:

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
9. Review the leakage findings, risk assessment, diagnostic model results, visualizations, and model comparison.
10. Download the structured analysis report in JSON format if needed.

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

As of October 28, 2026, the automated test suite contains 225 passing tests.

## Limitations

The application is intended to identify potential leakage risks rather than guarantee that all forms of data leakage will be detected.

Proxy leakage analysis relies on statistical relationships that require contextual interpretation. A strong feature-target relationship does not automatically establish leakage.

Preprocessing and cross-validation checks evaluate user-provided configuration settings rather than inspecting arbitrary external machine learning pipelines.

Diagnostic modeling focuses on numerical features and uses lightweight linear models. The application is intended for structured tabular datasets of manageable size and is not designed as a large-scale production monitoring system.

## Project Status

The core functionality has been implemented and tested. Final installation verification, end-to-end regression testing, and documentation refinement are in progress.

## Author

Jaiveer Kapadia