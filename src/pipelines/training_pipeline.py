"""
Training pipeline for end-to-end model training.
"""

import pandas as pd
from pathlib import Path
import mlflow
from src.data_ingestion.data_ingestion import load_data, validate_data
from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.feature_engineering.feature_engineering import FeatureEngineer
from src.models.train_model import train_model, hyperparameter_tuning
from src.utils.helpers import load_config, ensure_dir


class TrainingPipeline:
    """End-to-end training pipeline for term deposit prediction."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
    def run(self, data_path: str, target_column: str = 'y'):
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to raw data file
            target_column: Name of target column
            
        Returns:
            Dictionary containing trained model and metrics
        """
        # Load data
        print("Loading data...")
        separator = self.config.get('data', {}).get('separator', ',')
        df = load_data(data_path, sep=separator)
        
        # Validate data
        required_columns = [target_column]
        validate_data(df, required_columns)
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target variable
        y = y.map({'yes': 1, 'no': 0})
        
        # Preprocess data
        print("Preprocessing data...")
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        X = self.preprocessor.handle_missing_values(X, strategy='mean')
        X = self.preprocessor.encode_categorical(X, categorical_cols)
        X = self.preprocessor.scale_features(X, numeric_cols)
        
        # Feature engineering
        print("Engineering features...")
        # Add feature engineering steps as needed
        
        # Train model
        print("Training model...")
        model_type = self.config.get('model', {}).get('type', 'random_forest')
        params = self.config.get('model', {}).get('params', {})
        
        result = train_model(X, y, model_type=model_type, params=params)
        
        print(f"Training complete. CV F1 Score: {result['cv_f1_mean']:.4f}")
        
        return result


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    data_path = pipeline.config.get('data', {}).get('raw_path', 'data/raw/bank-full.csv')
    pipeline.run(data_path=data_path)
