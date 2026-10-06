"""
Model training module.
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from typing import Dict, Any


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                model_type: str = 'random_forest',
                params: Dict[str, Any] = None,
                experiment_name: str = 'term_deposit_prediction') -> Dict[str, Any]:
    """
    Train a machine learning model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('random_forest', 'gradient_boosting', 'logistic_regression')
        params: Model hyperparameters
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary containing trained model and metrics
    """
    mlflow.set_experiment(experiment_name)
    
    if params is None:
        params = {}
    
    # Initialize model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**params, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        
        # Log model
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        return {
            'model': model,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'run_id': mlflow.active_run().info.run_id
        }


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                         model_type: str = 'random_forest') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune
        
    Returns:
        Dictionary containing best model and parameters
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }
    
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    model = models[model_type]
    param_grid = param_grids[model_type]
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
