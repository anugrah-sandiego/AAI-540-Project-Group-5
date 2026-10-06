"""
Model prediction module.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Feature DataFrame
        
    Returns:
        Array of predictions
    """
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get prediction probabilities using a trained model.
    
    Args:
        model: Trained model object
        X: Feature DataFrame
        
    Returns:
        Array of prediction probabilities
    """
    return model.predict_proba(X)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = predict_model(model, X_test)
    y_proba = predict_proba(model, X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model object
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attribute")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
