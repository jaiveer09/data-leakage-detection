# System Architecture

## Architecture Overview

The system will use a modular architecture so that each major analysis task is handled by a separate component.

The main components are:

- Data loading and validation
- Dataset profiling
- Split configuration and validation
- Temporal leakage detection
- Group leakage detection
- Proxy/target leakage analysis
- Preprocessing and cross-validation validation
- Diagnostic modeling
- Warning and risk assessment
- Streamlit user interface
- Reporting and visualization

The Streamlit interface will collect user inputs and display results, while the analysis modules will contain the main project logic.

## System Workflow

1. User uploads a CSV dataset.
2. User selects the target/label column.
3. User optionally selects a timestamp column.
4. User optionally selects a group/entity column.
5. User provides the evaluation or split configuration.
6. The system validates the dataset and selected columns.
7. Dataset profiling is performed.
8. The system validates the train/test split configuration.
9. Temporal leakage checks are performed when timestamp data is available.
10. Group leakage checks are performed when group/entity data is available.
11. Proxy/target leakage analysis is performed.
12. Preprocessing and cross-validation configuration are checked.
13. Diagnostic models are used where appropriate.
14. Detected issues are converted into categorized warnings.
15. An overall leakage-risk assessment is produced.
16. Results, explanations, and visualizations are displayed to the user.

## Module Responsibilities

### data_loader.py

Responsible for:

- Loading CSV datasets
- Checking whether the dataset can be read successfully
- Validating that the selected target column exists
- Validating optional timestamp and group columns
- Returning the loaded and validated dataset for further analysis

### profiler.py

Responsible for:

- Dataset dimensions
- Column names
- Data types
- Missing-value information
- Basic numerical and categorical summaries
- Target-column summary

### split_validator.py

Responsible for:

- Representing or validating the selected evaluation configuration
- Checking train/test split settings
- Supporting random, time-based, and group-aware split validation where required
- Passing split information to other leakage modules

### temporal_detector.py

Responsible for:

- Analyzing the selected timestamp column
- Checking chronological consistency
- Detecting possible train/test temporal overlap
- Identifying situations where future information may influence past predictions

### group_detector.py

Responsible for:

- Analyzing the selected group/entity column
- Comparing entity membership across train and test data
- Detecting entities that appear in both splits

### proxy_detector.py

Responsible for:

- Analyzing relationships between features and the target
- Performing correlation-based analysis where appropriate
- Identifying features that may encode the target indirectly
- Supplying suspicious features for further diagnostic analysis

### pipeline_validator.py

Responsible for:

- Checking preprocessing configuration
- Checking whether preprocessing may have been performed before data splitting
- Checking cross-validation configuration
- Identifying evaluation practices that may introduce leakage

### diagnostic_model.py

Responsible for:

- Training simple diagnostic machine-learning models
- Evaluating prediction performance
- Comparing clean and potentially flawed configurations
- Providing feature-related diagnostic information

### risk_assessor.py

Responsible for:

- Receiving findings from all analysis modules
- Creating categorized warnings
- Assigning warning severity
- Combining findings into an overall leakage-risk assessment

### report_generator.py

Responsible for:

- Organizing analysis results
- Preparing structured output
- Preparing information used by the Streamlit interface
- Supporting visualizations and final analysis summaries

## Data Flow

The general data flow is:

CSV Dataset + User Configuration

→ Data Loading and Validation

→ Dataset Profiling

→ Split Validation

→ Leakage Detection Modules

- Temporal Leakage
- Group Leakage
- Proxy/Target Leakage

→ Pipeline and Evaluation Validation

→ Diagnostic Modeling

→ Warning and Risk Assessment

→ Reporting and Visualization

→ Streamlit Interface

## Inputs

The system will accept:

- CSV dataset
- Target/label column
- Optional timestamp column
- Optional group/entity column
- Evaluation and data-splitting configuration

## Outputs

The system will produce:

- Dataset profile
- Leakage warnings
- Warning categories
- Warning severity
- Explanations for detected risks
- Temporal leakage findings
- Group leakage findings
- Proxy/target leakage findings
- Pipeline and evaluation findings
- Diagnostic-model results
- Correlation analysis
- Relevant visualizations
- Overall leakage-risk assessment
- Structured analysis summary