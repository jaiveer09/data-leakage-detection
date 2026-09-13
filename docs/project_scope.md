# Project Scope

## Project Goal

Design and implement a practical system for detecting potential data leakage in machine learning pipelines.

The system will analyze datasets and evaluation configurations to identify leakage risks. It is intended to help users understand potential problems rather than guarantee perfect or fully automated leakage detection.

## Required Inputs

- CSV dataset
- Target/label column
- Optional timestamp column
- Optional group/entity column
- Evaluation and data-splitting configuration

## Required Analysis

### Dataset Handling and Profiling
- Load CSV datasets
- Validate dataset structure
- Identify data types
- Identify missing values
- Analyze basic distributions

### Temporal Leakage
- Analyze timestamp information
- Check chronological consistency
- Validate whether data splits preserve chronological order

### Group Leakage
- Analyze group/entity identifiers
- Detect overlapping entities between training and testing data

### Proxy/Target Leakage
- Analyze relationships between features and the target
- Perform correlation analysis
- Inspect potentially problematic features
- Use simple diagnostic models where appropriate

### Pipeline and Evaluation Validation
- Check data-splitting configuration
- Check for improper preprocessing
- Check for incorrect cross-validation usage
- Identify preprocessing performed outside the proper training/CV process

### Diagnostic Modeling
- Train simple diagnostic models, such as logistic regression
- Evaluate model behavior
- Compare performance under different configurations
- Help identify problematic features influencing predictions

## Required Outputs

- Detected leakage issues
- Categorized warnings
- Interpretable explanations
- Overall leakage-risk assessment
- Structured analysis/report
- Correlation analysis
- Relevant visualizations
- Correlation heatmaps
- Feature-importance plots
- Clean-versus-flawed pipeline comparisons

## Evaluation Requirements

- Test with real-world datasets
- Create synthetic datasets with controlled leakage scenarios
- Test temporal leakage detection
- Test group leakage detection
- Test proxy leakage analysis
- Validate correctness of detection modules
- Evaluate consistency of results
- Compare model performance between clean and flawed configurations
- Demonstrate how leakage can inflate evaluation results

## Development Constraints

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib/seaborn
- Streamlit
- VS Code or similar IDE
- Run locally
- Support manageable/moderate-sized tabular datasets
- Keep implementation lightweight
- Do not require specialized hardware
- Do not use deep-learning models
- Keep detection modules modular and maintainable
- Produce consistent results for the same input
- Provide a simple and intuitive interface

## Out of Scope

The following are not part of the approved proposal and should not be added unless there is a justified project change:

- Deep-learning models
- Cloud deployment
- Specialized/high-performance computing
- Mobile application
- User accounts or authentication
- Database backend
- LLM-based leakage detection
- AutoML system
- Guaranteed detection of every possible form of data leakage
- Fully automatic domain understanding
- Additional major leakage categories beyond the approved scope

The system should identify and explain potential leakage risks. It should not claim that every warning definitively proves that leakage exists.