# Term Deposit Prediction - Project Documentation

**Group 5 Members:**
- Dhrub Satyam
- Anugrah Rastogi
- Aishwarya Gulhane

**Project:** AAI-540-Project-Group-5  
**Date:** October 6, 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Background](#project-background)
3. [Technical Architecture](#technical-architecture)
4. [Dataset Information](#dataset-information)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Pipeline Workflow](#pipeline-workflow)
8. [Model Training](#model-training)
9. [Model Evaluation](#model-evaluation)
10. [MLOps with MLflow](#mlops-with-mlflow)
11. [AWS SageMaker Deployment](#aws-sagemaker-deployment)
12. [Usage Examples](#usage-examples)
13. [Configuration](#configuration)
14. [Troubleshooting](#troubleshooting)
15. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a production-oriented machine learning system to predict customer subscription to term deposits for a Portuguese banking institution. The system follows MLOps best practices with a complete end-to-end pipeline including data ingestion, preprocessing, feature engineering, model training, evaluation, and experiment tracking.

### Key Features

- **Modular Architecture**: Separated components for data ingestion, preprocessing, feature engineering, and modeling
- **MLOps Integration**: MLflow for experiment tracking and model registry
- **Multiple Model Support**: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, each with three class-imbalance strategies
- **Reproducible Pipeline**: Configuration-driven workflow with version control
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC

---

## Project Background

### Business Problem

Banking institutions conduct direct marketing campaigns (phone calls) to promote term deposit subscriptions. The challenge is to identify customers most likely to subscribe, optimizing marketing resources and improving campaign efficiency.

### Dataset

The project uses the **UCI Bank Marketing Dataset**, a publicly available dataset containing information from direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (yes/no) to a term deposit (variable `y`).

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Dataset Size:** 45,211 records with 17 features

---

## Technical Architecture

### Technology Stack

- **Language:** Python 3.10+
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn
- **MLOps:** MLflow
- **Visualization:** matplotlib, seaborn
- **Configuration:** PyYAML
- **Containerization:** Docker
- **Testing:** pytest

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                      │
│  - Load semicolon-separated CSV / Excel / Parquet             │
│  - Schema & data-quality validation (columns, types, nulls,  │
│    target values, duplicates, 'unknown' rates)               │
│  - Stratified, shuffled 70 / 15 / 15 train/val/test split    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│        Feature Engineering + Preprocessing (sklearn Pipeline) │
│  - Drop 'duration' (post-call leakage)                       │
│  - Domain features: previously_contacted, prior_success,     │
│    log_campaign, log_previous, negative_balance              │
│  - One-hot encode nominal categoricals, scale numerics       │
│  - Fitted on the training split only                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Layer                       │
│  - Logistic Regression, Decision Tree, Random Forest,        │
│    Gradient Boosting (HistGradientBoosting)                  │
│  - Class-imbalance strategies: none / class_weight / SMOTE   │
│  - Shuffled stratified 5-fold CV, selection by PR-AUC        │
│  - Bounded RandomizedSearchCV on the best candidate          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Model Evaluation Layer                       │
│  - Decision threshold tuned on validation (max F1)           │
│  - Test: precision, recall, F1, ROC-AUC, PR-AUC, confusion   │
│  - Business: conversion & lift in top 10 / 20 / 30 %         │
│  - Quality gates (ROC-AUC, lift@10) fail the pipeline        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                MLOps Layer (MLflow, SQLite backend)           │
│  - Nested run per model/strategy, comparison table           │
│  - Parameters, metrics, data-quality report, model artifact  │
│  - models/model.joblib + model_metadata.json for serving     │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Information

### Data Source

**File:** `data/raw/bank-full.csv`  
**Format:** Semicolon-separated CSV  
**Records:** 45,211  
**Features:** 17 + 1 target variable

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Age of the client |
| job | categorical | Type of job (admin, technician, services, etc.) |
| marital | categorical | Marital status (married, single, divorced) |
| education | categorical | Education level (primary, secondary, tertiary) |
| default | categorical | Has credit in default? (yes, no) |
| balance | numeric | Average yearly balance in euros |
| housing | categorical | Has housing loan? (yes, no) |
| loan | categorical | Has personal loan? (yes, no) |
| contact | categorical | Contact communication type |
| day | numeric | Last contact day of the month |
| month | categorical | Last contact month of year |
| duration | numeric | Last contact duration, in seconds |
| campaign | numeric | Number of contacts performed during campaign |
| pdays | numeric | Days passed since last contact |
| previous | numeric | Number of contacts performed before this campaign |
| poutcome | categorical | Outcome of previous marketing campaign |
| y | target | Has the client subscribed? (yes, no) |

### Data Characteristics

- **Class Imbalance:** The dataset is imbalanced with approximately 11.7% positive class (subscribers)
- **Chronological Ordering:** Rows are ordered by contact date (May 2008 – Nov 2010) and the positive rate rises from ~3% in the first fifth of the file to ~32% in the last fifth. Splits and CV folds must be shuffled; this is also a real drift signal to monitor in production.
- **Missing Values:** No nulls, but some categoricals contain "unknown" (poutcome 82%, contact 29%, education 4%, job <1%), which is kept as its own category
- **Leakage:** `duration` is only known after the call, so it is excluded from the pre-contact model
- **Data Types:** Mix of numerical and categorical features

---

## Project Structure

```
AAI-540-Project-Group-5/
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── bank-full.csv       # Full dataset (45,211 records)
│   │   ├── bank.csv            # Sample dataset (10%)
│   │   └── bank-names.txt      # Feature descriptions
│   ├── processed/              # train/val/test Parquet splits (generated)
│   └── external/               # External reference data
├── notebooks/
│   ├── exploratory_data_analysis.ipynb    # EDA notebook
│   ├── model_development.ipynb           # Model training notebook
│   └── model_evaluation.ipynb             # Model evaluation notebook
├── src/
│   ├── data_ingestion/
│   │   └── data_ingestion.py              # Data loading and validation
│   ├── data_preprocessing/
│   │   └── data_preprocessing.py          # Data cleaning and transformation
│   ├── feature_engineering/
│   │   └── feature_engineering.py         # Feature creation and selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py                 # Model training with MLflow
│   │   └── predict_model.py               # Model prediction and evaluation
│   ├── utils/
│   │   └── helpers.py                     # Helper utilities
│   ├── pipelines/
│   │   └── training_pipeline.py            # End-to-end training pipeline
│   └── aws/
│       └── sagemaker/
│           ├── data_upload.py              # S3 data upload
│           ├── feature_store.py            # SageMaker Feature Store
│           ├── training.py                 # SageMaker training jobs
│           ├── deployment.py               # Endpoint deployment
│           ├── batch_inference.py          # Batch transform jobs
│           ├── model_registry.py           # Model Registry
│           ├── monitoring.py               # Model monitoring
│           ├── infrastructure_monitoring.py # CloudWatch dashboards
│           ├── rollback.py                 # Rollback utilities
│           └── scripts/
│               ├── train.py                # Training script
│               ├── inference.py            # Inference script
│               └── batch_transform.py      # Batch transform script
├── models/                                # model.joblib + model_metadata.json (generated)
├── mlflow.db                              # MLflow tracking store (generated)
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py         # Preprocessing tests
│   └── test_model_training.py             # Model training tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml                     # GitHub Actions CI/CD pipeline
├── requirements.txt                       # Python dependencies
├── Dockerfile                            # Container configuration
├── config.yaml                           # Project configuration
├── .env                                  # Environment variables
└── README.md                             # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/anugrah-sandiego/AAI-540-Project-Group-5.git
   cd AAI-540-Project-Group-5
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pandas, sklearn, mlflow; print('Dependencies installed successfully')"
   ```

5. **Set Environment Variables (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your specific configurations
   ```

---

## Pipeline Workflow

### Step 1: Data Ingestion

**Location:** `src/data_ingestion/data_ingestion.py`

**Functions:**
- `load_data(file_path, sep=',')`: Loads CSV, Excel or Parquet files
- `validate_schema(df)`: Checks expected columns, numeric types, nulls, target values; returns a data-quality report
- `split_data(X, y, val_size, test_size, random_state)`: Stratified, shuffled train/val/test split

```python
from src.data_ingestion.data_ingestion import load_data, validate_schema, split_data

df = load_data('data/raw/bank-full.csv', sep=';')
report = validate_schema(df)
```

### Step 2: Preprocessing and Feature Engineering

**Locations:** `src/data_preprocessing/data_preprocessing.py`, `src/feature_engineering/feature_engineering.py`

- `encode_target(y)`: Maps yes/no to 1/0
- `build_preprocessor(numeric_cols, categorical_cols)`: `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')`
- `add_domain_features(df)`: Stateless, row-wise pre-contact features (`DOMAIN_FEATURES`)

These are not applied to the data up front. They are steps inside the model pipeline, so they are fitted on training folds only and the saved model accepts raw columns.

### Step 3: Model Training

**Location:** `src/models/train_model.py`

- `build_model(model_type, numeric_cols, categorical_cols, imbalance_strategy)`: Builds an imblearn `Pipeline` of features → preprocess → [SMOTE] → classifier
- `compare_models(...)`: Cross-validates every model × imbalance strategy and logs each as a nested MLflow run
- `hyperparameter_tuning(...)`: Bounded `RandomizedSearchCV` optimising PR-AUC

**Models:** `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`
**Imbalance strategies:** `none`, `class_weight` (balanced loss), `smote` (oversampling inside CV folds only)

### Step 4: Model Evaluation

**Location:** `src/models/predict_model.py`

- `predict_proba(model, X)`: Positive-class probabilities
- `predict_model(model, X, threshold)`: Class predictions at a tuned threshold
- `find_best_threshold(y_val, proba)`: Threshold maximising F1 on validation data
- `lift_at_k(y, proba, [10, 20, 30])`: Conversion rate and lift in the top-k% scored customers
- `evaluate_model(model, X, y, threshold)`: Technical + business metrics
- `get_feature_importance(model, feature_names)`: Works with a bare estimator or the pipeline

---

## Model Training

### Running the Training Pipeline

```bash
python -m src.pipelines.training_pipeline
```

**Pipeline Steps:**
1. Load configuration from `config.yaml`
2. Load and validate data, log the data-quality report
3. Drop excluded (leaky) features and encode the target
4. Stratified 70/15/15 split; save splits to `data/processed/*.parquet`
5. Cross-validate 4 models × 3 imbalance strategies on the training split
6. Tune the best candidate (by CV PR-AUC)
7. Choose the decision threshold on the validation split
8. Evaluate once on the test split (technical + business metrics)
9. Check quality gates; fail the run if they are not met
10. Log everything to MLflow and save `models/model.joblib` and `models/model_metadata.json`

### Results (test split, 6,782 customers, `duration` excluded)

Cross-validation on the training split (mean of 5 shuffled folds, default 0.5 threshold for F1):

| Model | Strategy | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|---|
| Logistic Regression | class_weight | 0.766 | 0.406 | 0.378 | 0.624 |
| Decision Tree | class_weight | 0.754 | 0.354 | 0.415 | 0.553 |
| Random Forest | none | **0.791** | **0.445** | 0.290 | 0.185 |
| Random Forest | class_weight | 0.790 | 0.438 | 0.471 | 0.515 |
| Random Forest | smote | 0.783 | 0.419 | 0.439 | 0.399 |
| Gradient Boosting | class_weight | 0.794 | 0.442 | 0.455 | 0.617 |

Selected model: tuned Random Forest (`max_depth=16, max_features=0.5, min_samples_leaf=8, n_estimators=314`), decision threshold 0.169 chosen on validation.

| Metric | Threshold 0.50 | Tuned threshold 0.169 |
|---|---|---|
| Precision | 0.620 | 0.432 |
| Recall | 0.216 | 0.532 |
| F1 | 0.320 | **0.477** |

ROC-AUC 0.793, PR-AUC 0.440.

| Targeted segment | Conversion | Lift vs. random outreach (11.7%) |
|---|---|---|
| Top 10% | 49.3% | 4.22× |
| Top 20% | 35.4% | 3.03× |
| Top 30% | 26.8% | 2.29× |

**Findings on class imbalance:** class weights and SMOTE raise F1 at the default 0.5 threshold because they shift predicted probabilities upwards, but they do not improve ranking quality (ROC-AUC / PR-AUC). SMOTE was consistently slightly worse than class weights. Tuning the decision threshold on validation data gives the same F1 gain without distorting probabilities, so the selected model uses no resampling plus a tuned threshold. The earlier baseline F1 of ~0.11 was mostly an artefact of unshuffled CV folds over chronologically ordered data, not of class imbalance.

---

## Model Evaluation

### Evaluation Metrics

- **ROC-AUC / PR-AUC:** Threshold-independent ranking quality; PR-AUC is the model-selection metric because it focuses on the minority class
- **Precision / Recall / F1:** Reported at the default and the tuned threshold
- **Confusion matrix:** TP / FP / TN / FN counts logged to MLflow
- **Conversion and lift @ 10/20/30%:** Business value of targeting the highest-scored customers
- Accuracy is reported but never used for selection (an all-"no" model scores 88%)

### Quality Gates

Configured in `config.yaml` under `training.quality_gates` (currently ROC-AUC ≥ 0.75 and lift@10 ≥ 2.5). The pipeline raises an error and tags the MLflow run `quality_gates_passed=False` if the test metrics fall below them.

---

## MLOps with MLflow

### MLflow Tracking

**Start MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Access:** http://localhost:5000

### Experiment Tracking

Each pipeline run creates a parent run `training_pipeline` with:
- Nested runs for every model × imbalance strategy (CV metrics)
- `model_comparison.json` table and `data_quality_report.json`
- Selected model type, strategy, hyperparameters and decision threshold
- Validation and test metrics, including lift and conversion
- The fitted pipeline with signature and input example

### Loading a Trained Model

```python
import json, joblib

model = joblib.load('models/model.joblib')
threshold = json.load(open('models/model_metadata.json'))['decision_threshold']

proba = model.predict_proba(raw_customers_df)[:, 1]   # raw columns, no preprocessing needed
predicted = proba >= threshold
```

---

## AWS SageMaker Deployment

This project includes comprehensive AWS SageMaker integration for production deployment. The implementation includes:

### SageMaker Components

1. **Data Upload to S3** - Upload local data to S3 for SageMaker processing
2. **Feature Store** - SageMaker Feature Groups for storing and retrieving features
3. **Training Jobs** - Managed training jobs on SageMaker
4. **Model Deployment** - Real-time endpoint deployment
5. **Batch Inference** - Batch transform jobs for bulk predictions
6. **Model Registry** - Model versioning and approval workflow
7. **Model Monitoring** - Data capture and quality monitoring
8. **Infrastructure Monitoring** - CloudWatch dashboards for system monitoring
9. **CI/CD Pipeline** - GitHub Actions for automated deployment

### Prerequisites

Before deploying to SageMaker, ensure you have:

- AWS account with appropriate IAM permissions
- AWS credentials configured in `.env` file:
  ```bash
  AWS_ACCOUNT_ID=your_account_id
  AWS_REGION=us-east-1
  AWS_ROLE_ARN=arn:aws:iam::your_account_id:role/your_role
  SAGEMAKER_OUTPUT_PATH=s3://your-bucket/path
  ```
- SageMaker execution role with necessary permissions
- S3 bucket for data storage

### AWS SageMaker Setup

1. **Install AWS Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS Credentials**
   ```bash
   # Edit .env file with your AWS credentials
   ```

3. **Upload Data to S3**
   ```bash
   python -m src.aws.sagemaker.data_upload
   ```

4. **Create Feature Store**
   ```bash
   python -m src.aws.sagemaker.feature_store
   ```

5. **Run SageMaker Training**
   ```bash
   python -m src.aws.sagemaker.training
   ```

6. **Register Model in Model Registry**
   ```bash
   python -m src.aws.sagemaker.model_registry
   ```

7. **Deploy Model to Endpoint**
   ```bash
   python -m src.aws.sagemaker.deployment
   ```

8. **Set Up Model Monitoring**
   ```bash
   python -m src.aws.sagemaker.monitoring
   ```

9. **Create Infrastructure Monitoring Dashboards**
   ```bash
   python -m src.aws.sagemaker.infrastructure_monitoring
   ```

### CI/CD Pipeline

The project includes a GitHub Actions CI/CD pipeline (`.github/workflows/ci-cd.yml`) that:

- Runs unit tests on every push and pull request
- Builds and pushes Docker images
- Deploys to SageMaker on main branch pushes
- Includes rollback mechanism on deployment failure
- Supports manual batch inference job triggers

**Required GitHub Secrets:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ACCOUNT_ID`
- `AWS_ROLE_ARN`
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`



### Monitoring Dashboards

The system creates three CloudWatch dashboards:

1. **ml-system-monitoring** - Comprehensive system-wide monitoring
2. **training-jobs-monitoring** - Training job specific metrics
3. **endpoint-monitoring** - Endpoint performance metrics

Each dashboard includes:
- Resource utilization (CPU, memory, GPU)
- Latency metrics
- Error rates
- Request counts
- S3 operations
- Feature Store performance

### Batch Inference

To run batch inference on a dataset:

```bash
python -m src.aws.sagemaker.batch_inference
```

This will:
- Create a transformer from the registered model
- Process input data from S3
- Output predictions to S3
- Monitor job completion

### Rollback

In case of deployment failure, the CI/CD pipeline automatically triggers rollback:

```bash
python -m src.aws.sagemaker.rollback
```

This reverts the endpoint to the last approved model version.

---

## Usage Examples

### Example 1: Quick Start with Training Pipeline

```bash
python -m src.pipelines.training_pipeline
```

### Example 2: Custom Training in Notebook

```python
from src.data_ingestion.data_ingestion import load_data, split_data, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.data_preprocessing.data_preprocessing import encode_target
from src.feature_engineering.feature_engineering import DOMAIN_FEATURES
from src.models.train_model import build_model
from src.models.predict_model import find_best_threshold, evaluate_model, predict_proba

df = load_data('data/raw/bank-full.csv', sep=';')
X, y = df.drop(columns=['y', 'duration']), encode_target(df['y'])
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

numeric_cols = [c for c in NUMERIC_COLUMNS if c != 'duration'] + DOMAIN_FEATURES
model = build_model('gradient_boosting', numeric_cols, CATEGORICAL_COLUMNS, 'class_weight')
model.fit(X_train, y_train)

threshold = find_best_threshold(y_val, predict_proba(model, X_val))['threshold']
print(evaluate_model(model, X_test, y_test, threshold))
```

### Example 3: Running Tests

```bash
pytest tests/
```

### Example 4: Docker Deployment

```bash
# Build image
docker build -t term-deposit-prediction .

# Run container
docker run term-deposit-prediction
```

---

## Configuration

### config.yaml

The main configuration file controls the data split, excluded features, candidate models, imbalance strategies, tuning budget, threshold metric, quality gates and MLflow settings:

```yaml
data:
  raw_path: data/raw/bank-full.csv
  separator: ";"
  val_size: 0.15
  test_size: 0.15
  excluded_features: [duration]

training:
  cv_folds: 5
  selection_metric: average_precision
  candidate_models: [logistic_regression, decision_tree, random_forest, gradient_boosting]
  imbalance_strategies: [none, class_weight, smote]
  tuning_iterations: 20
  threshold_metric: f1
  quality_gates:
    roc_auc: 0.75
    lift_at_10: 2.5

mlflow:
  experiment_name: term_deposit_prediction
  tracking_uri: sqlite:///mlflow.db
```

### Environment Variables (.env)

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=term_deposit_prediction
DATA_RAW_PATH=data/raw
DATA_PROCESSED_PATH=data/processed
DATA_EXTERNAL_PATH=data/external
MODEL_TYPE=random_forest
RANDOM_STATE=42
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from the project root as a module:
```bash
python -m src.pipelines.training_pipeline
```

#### Issue 2: MLflow File-Store Error

**Error:** `The filesystem tracking backend (e.g., './mlruns') is in maintenance mode`

**Solution:** MLflow 3 requires a database backend. The project uses `sqlite:///mlflow.db` (set in `config.yaml` and `.env`).

#### Issue 3: Low F1 Score

**Issue:** F1 score around 0.1 when cross-validating on the full file

**Cause:** `bank-full.csv` is sorted by date, and unshuffled CV folds train on one period and test on another with a very different positive rate.

**Solution:** Use shuffled `StratifiedKFold` (already done in `compare_models` / `hyperparameter_tuning`) and tune the decision threshold on validation data.

#### Issue 4: Data Loading Error

**Error:** CSV parsing errors

**Solution:** Ensure the correct separator is specified in `config.yaml`. The bank dataset uses semicolon separator (`;`).

#### Issue 5: Memory Issues

**Issue:** Out of memory errors with large datasets

**Solution:**
- Use the smaller `bank.csv` (10% sample) for testing
- Reduce model complexity (fewer trees, lower depth)
- Process data in chunks

---

## Future Improvements

### Model Improvements

1. **Class Imbalance Handling**
   - Try focal loss or probability calibration (Platt / isotonic) for better-calibrated scores
   - Cost-sensitive threshold based on the actual cost of a call vs. the value of a deposit

2. **Advanced Feature Engineering**
   - Create domain-specific features (e.g., customer segmentation)
   - Add temporal features from date columns
   - Implement automated feature selection

3. **Model Ensemble**
   - Combine multiple models using voting or stacking
   - Implement Bayesian optimization for hyperparameter tuning

### Pipeline Improvements

1. **Data Validation**
   - Add data drift detection
   - Implement data quality checks
   - Add automated data profiling

2. **Model Monitoring**
   - Implement model performance monitoring
   - Add alerting for model degradation
   - Track prediction distribution over time

3. **Deployment**
   - Create REST API for model inference
   - Implement batch prediction pipeline
   - Add A/B testing framework

### Infrastructure

1. **CI/CD Pipeline**
   - Automate testing with GitHub Actions
   - Implement automated model training
   - Add model versioning

2. **Scalability**
   - Implement distributed training
   - Add support for streaming data
   - Optimize for cloud deployment

---

## References

### Dataset Citation

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

### Related Papers

Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31.

### Useful Resources

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [MLflow Documentation](https://mlflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## Contact

**Group 5 Members:**
- Dhrub Satyam
- Anugrah Rastogi
- Aishwarya Gulhane

**Project:** AAI-540-Project-Group-5  
**Repository:** https://github.com/anugrah-sandiego/AAI-540-Project-Group-5

---

*Last Updated: October 6, 2026*