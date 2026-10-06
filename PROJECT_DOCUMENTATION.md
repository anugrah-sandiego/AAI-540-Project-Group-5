# Term Deposit Prediction - Project Documentation

**Author:** Dhrub Satyam  
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
11. [Usage Examples](#usage-examples)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)
14. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a production-oriented machine learning system to predict customer subscription to term deposits for a Portuguese banking institution. The system follows MLOps best practices with a complete end-to-end pipeline including data ingestion, preprocessing, feature engineering, model training, evaluation, and experiment tracking.

### Key Features

- **Modular Architecture**: Separated components for data ingestion, preprocessing, feature engineering, and modeling
- **MLOps Integration**: MLflow for experiment tracking and model registry
- **Multiple Model Support**: Random Forest, Gradient Boosting, and Logistic Regression
- **Reproducible Pipeline**: Configuration-driven workflow with version control
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC

---

## Project Background

### Business Problem

Banking institutions conduct direct marketing campaigns (phone calls) to promote term deposit subscriptions. The challenge is to identify customers most likely to subscribe, optimizing marketing resources and improving campaign efficiency.

### Dataset

The project uses the **UCI Bank Marketing Dataset**, a publicly available dataset containing information from direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (yes/no) to a term deposit (variable `y`).

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Dataset Size:** 45,213 records with 17 features

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
│  - Load data from various formats (CSV, Excel)               │
│  - Validate data integrity                                   │
│  - Handle different separators (comma, semicolon)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Preprocessing Layer                     │
│  - Handle missing values                                     │
│  - Encode categorical variables                             │
│  - Scale numerical features                                  │
│  - Store fitted transformers for reproducibility             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering Layer                     │
│  - Create interaction features                              │
│  - Generate polynomial features                             │
│  - Feature selection (K-Best, Mutual Information)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Layer                       │
│  - Random Forest Classifier                                   │
│  - Gradient Boosting Classifier                              │
│  - Logistic Regression                                       │
│  - Hyperparameter tuning with GridSearchCV                    │
│  - Cross-validation                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Model Evaluation Layer                       │
│  - Accuracy, Precision, Recall, F1-Score                     │
│  - ROC-AUC analysis                                          │
│  - Confusion matrix                                          │
│  - Feature importance analysis                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      MLOps Layer (MLflow)                     │
│  - Experiment tracking                                       │
│  - Model registry                                            │
│  - Parameter logging                                         │
│  - Metric logging                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Information

### Data Source

**File:** `data/raw/bank-full.csv`  
**Format:** Semicolon-separated CSV  
**Records:** 45,213  
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
- **Missing Values:** Some categorical features have "unknown" values
- **Data Types:** Mix of numerical and categorical features

---

## Project Structure

```
AAI-540-Project-Group-5/
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── bank-full.csv       # Full dataset (45,213 records)
│   │   ├── bank.csv            # Sample dataset (10%)
│   │   └── bank-names.txt      # Feature descriptions
│   ├── processed/              # Preprocessed data (generated)
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
│   └── pipelines/
│       └── training_pipeline.py            # End-to-end training pipeline
├── mlruns/                                # MLflow experiment tracking
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py         # Preprocessing tests
│   └── test_model_training.py             # Model training tests
├── requirements.txt                       # Python dependencies
├── Dockerfile                            # Container configuration
├── config.yaml                           # Project configuration
├── .env                                  # Environment variables
├── README.md                             # Project overview
└── PROJECT_DOCUMENTATION.md              # This file
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
- `load_data(file_path, sep=',')`: Loads data from CSV or Excel files
- `validate_data(df, required_columns)`: Validates required columns exist

**Usage:**
```python
from src.data_ingestion.data_ingestion import load_data

# Load data with semicolon separator
df = load_data('data/raw/bank-full.csv', sep=';')
```

### Step 2: Data Preprocessing

**Location:** `src/data_preprocessing/data_preprocessing.py`

**Class:** `DataPreprocessor`

**Methods:**
- `handle_missing_values(df, strategy='mean')`: Handles missing values
- `encode_categorical(df, categorical_cols)`: Encodes categorical variables
- `scale_features(df, numeric_cols, fit=True)`: Scales numerical features

**Usage:**
```python
from src.data_preprocessing.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X = preprocessor.handle_missing_values(X, strategy='mean')
X = preprocessor.encode_categorical(X, categorical_cols)
X = preprocessor.scale_features(X, numeric_cols)
```

### Step 3: Feature Engineering

**Location:** `src/feature_engineering/feature_engineering.py`

**Class:** `FeatureEngineer`

**Methods:**
- `create_interaction_features(df, feature_pairs)`: Creates interaction features
- `create_polynomial_features(df, cols, degree=2)`: Creates polynomial features
- `select_features(X, y, method='k_best', k=10)`: Selects important features

**Usage:**
```python
from src.feature_engineering.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer()
X = feature_engineer.create_interaction_features(X, [('age', 'balance')])
X_selected, selected_features = feature_engineer.select_features(X, y, k=10)
```

### Step 4: Model Training

**Location:** `src/models/train_model.py`

**Functions:**
- `train_model(X_train, y_train, model_type, params)`: Trains model with MLflow tracking
- `hyperparameter_tuning(X_train, y_train, model_type)`: Performs hyperparameter tuning

**Supported Models:**
- Random Forest
- Gradient Boosting
- Logistic Regression

**Usage:**
```python
from src.models.train_model import train_model

result = train_model(
    X_train, y_train,
    model_type='random_forest',
    params={'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
)
```

### Step 5: Model Evaluation

**Location:** `src/models/predict_model.py`

**Functions:**
- `predict_model(model, X)`: Makes predictions
- `predict_proba(model, X)`: Gets prediction probabilities
- `evaluate_model(model, X_test, y_test)`: Evaluates model performance
- `get_feature_importance(model, feature_names)`: Gets feature importance

**Usage:**
```python
from src.models.predict_model import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(metrics)
# Output: {'accuracy': 0.90, 'precision': 0.65, 'recall': 0.35, 'f1': 0.45, 'roc_auc': 0.85}
```

---

## Model Training

### Running the Training Pipeline

**Command:**
```bash
PYTHONPATH=/Users/thedatayogi/CascadeProjects/AAI-540-Project-Group-5 python src/pipelines/training_pipeline.py
```

**Pipeline Steps:**
1. Load configuration from `config.yaml`
2. Load data from configured path
3. Validate data integrity
4. Split features and target
5. Encode target variable (yes/no → 1/0)
6. Preprocess data (handle missing values, encode categoricals, scale features)
7. Train model with configured parameters
8. Log metrics and model to MLflow
9. Return training results

### Model Configuration

**Configuration File:** `config.yaml`

```yaml
model:
  type: random_forest
  params:
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5
    random_state: 42
```

### Hyperparameter Tuning

**Usage:**
```python
from src.models.train_model import hyperparameter_tuning

tuning_result = hyperparameter_tuning(X_train, y_train, model_type='random_forest')
print(f"Best parameters: {tuning_result['best_params']}")
print(f"Best CV score: {tuning_result['best_score']}")
```

---

## Model Evaluation

### Evaluation Metrics

The project uses the following metrics:

- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve

### Running Evaluation Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook notebooks/
   ```

2. Open `model_evaluation.ipynb`

3. Update the run ID with your MLflow run ID

4. Run all cells to see evaluation results

### Evaluation Outputs

- Confusion matrix visualization
- ROC curve
- Precision-Recall curve
- Feature importance plot
- Threshold analysis

---

## MLOps with MLflow

### MLflow Tracking

**Start MLflow UI:**
```bash
mlflow ui
```

**Access:** http://localhost:5000

### Experiment Tracking

All model training runs are automatically tracked with:
- Parameters (model type, hyperparameters)
- Metrics (CV F1 score, accuracy, etc.)
- Artifacts (trained model)
- Run metadata (timestamp, duration)

### Loading a Trained Model

```python
import mlflow
import mlflow.sklearn

# Load model from MLflow
run_id = "your_run_id"
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
```

---

## Usage Examples

### Example 1: Quick Start with Training Pipeline

```bash
# Run the complete pipeline
PYTHONPATH=/path/to/project python src/pipelines/training_pipeline.py
```

### Example 2: Custom Training in Notebook

```python
import sys
sys.path.append('/path/to/project')

from src.data_ingestion.data_ingestion import load_data
from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.models.train_model import train_model

# Load data
df = load_data('data/raw/bank-full.csv', sep=';')

# Preprocess
preprocessor = DataPreprocessor()
X = df.drop(columns=['y'])
y = df['y'].map({'yes': 1, 'no': 0})

# Encode and scale
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
X = preprocessor.handle_missing_values(X)
X = preprocessor.encode_categorical(X, categorical_cols)
X = preprocessor.scale_features(X, numeric_cols)

# Train
result = train_model(X, y, model_type='random_forest')
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

The main configuration file controls:

```yaml
# Data configuration
data:
  raw_path: data/raw/bank-full.csv
  separator: ";"
  processed_path: data/processed
  test_size: 0.2
  random_state: 42

# Preprocessing configuration
preprocessing:
  missing_value_strategy: mean
  scale_features: true
  encode_categorical: true

# Feature engineering configuration
feature_engineering:
  create_interactions: false
  create_polynomial: false
  feature_selection: false
  k_best_features: 10

# Model configuration
model:
  type: random_forest
  params:
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5
    random_state: 42

# MLflow configuration
mlflow:
  experiment_name: term_deposit_prediction
  tracking_uri: ./mlruns
```

### Environment Variables (.env)

```bash
# MLflow configuration
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=term_deposit_prediction

# Data paths
DATA_RAW_PATH=data/raw
DATA_PROCESSED_PATH=data/processed
DATA_EXTERNAL_PATH=data/external

# Model configuration
MODEL_TYPE=random_forest
RANDOM_STATE=42
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Set PYTHONPATH before running scripts:
```bash
PYTHONPATH=/path/to/project python src/pipelines/training_pipeline.py
```

#### Issue 2: MLflow Trusted Types Error

**Error:** `Untrusted types found in the file: ['sklearn.tree._tree.Tree']`

**Solution:** The code already handles this by setting `skops_trusted_types=["sklearn.tree._tree.Tree"]` in the model logging function.

#### Issue 3: Low F1 Score

**Issue:** F1 score around 0.1-0.2

**Causes:**
- Class imbalance in dataset (only ~11.7% positive class)
- Default threshold of 0.5 may not be optimal

**Solutions:**
- Use class weights in model parameters
- Apply SMOTE or other oversampling techniques
- Adjust decision threshold based on precision-recall trade-off
- Try different models or hyperparameters

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
   - Implement SMOTE for oversampling
   - Add class weights to models
   - Try focal loss for imbalanced classification

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

**Author:** Dhrub Satyam  
**Project:** AAI-540-Project-Group-5  
**Repository:** https://github.com/anugrah-sandiego/AAI-540-Project-Group-5

---

*Last Updated: October 6, 2026*
