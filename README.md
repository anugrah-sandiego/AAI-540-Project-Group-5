# AAI-540-Project-Group-5

## Project Overview

This project focuses on building a production-oriented machine learning system to predict customer subscription to a term deposit. The system implements a complete MLOps workflow including data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment tracking.

## Project Background

The banking sector aims to improve marketing campaign efficiency by identifying customers most likely to subscribe to term deposits. This ML system helps optimize marketing resources by predicting subscription probability based on customer demographics, campaign history, and economic indicators.

## Technical Background

- **Machine Learning Pipeline**: End-to-end ML workflow with modular components
- **MLOps**: MLflow integration for experiment tracking and model registry
- **Data Management**: Structured data storage with raw, processed, and external data directories
- **Model Types**: Random Forest, Gradient Boosting, and Logistic Regression
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Project Goals

1. Build a production-ready ML pipeline for term deposit prediction
2. Implement comprehensive data preprocessing and feature engineering
3. Train and evaluate multiple ML models with hyperparameter tuning
4. Establish MLOps practices with MLflow for experiment tracking
5. Create reproducible and maintainable code structure

## Project Structure

```
AAI-540-Project-Group-5/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Preprocessed data
│   └── external/         # External reference data
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_development.ipynb
│   └── model_evaluation.ipynb
├── src/
│   ├── data_ingestion/   # Data loading and validation
│   ├── data_preprocessing/  # Data cleaning and transformation
│   ├── feature_engineering/ # Feature creation and selection
│   ├── models/           # Model training and prediction
│   ├── utils/            # Helper utilities
│   └── pipelines/        # End-to-end training pipeline
├── mlruns/               # MLflow experiment tracking
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── config.yaml           # Project configuration
└── .env                  # Environment variables
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anugrah-sandiego/AAI-540-Project-Group-5.git
cd AAI-540-Project-Group-5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Usage

#### Run Training Pipeline
```bash
python src/pipelines/training_pipeline.py
```

#### Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

#### Run Tests
```bash
pytest tests/
```

#### Docker Build
```bash
docker build -t term-deposit-prediction .
docker run term-deposit-prediction
```

## MLflow Tracking

Start MLflow UI to view experiments:
```bash
mlflow ui
```

Access at: http://localhost:5000

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Preprocessing strategies
- Model parameters
- Feature engineering options

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

This project is part of AAI-540 coursework.
