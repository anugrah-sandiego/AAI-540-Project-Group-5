"""
Tests for model training module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train_model import train_model


def test_train_model():
    """Test model training function."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train = pd.DataFrame(X)
    y_train = pd.Series(y)
    
    # Train model
    result = train_model(
        X_train, y_train,
        model_type='random_forest',
        params={'n_estimators': 10, 'random_state': 42},
        experiment_name='test_experiment'
    )
    
    assert 'model' in result
    assert 'cv_f1_mean' in result
    assert 'run_id' in result
    assert result['cv_f1_mean'] >= 0
    assert result['cv_f1_mean'] <= 1
