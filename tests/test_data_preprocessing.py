"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing.data_preprocessing import DataPreprocessor


def test_handle_missing_values_mean():
    """Test missing value handling with mean strategy."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8]
    })
    
    result = preprocessor.handle_missing_values(df, strategy='mean')
    
    assert result.isnull().sum().sum() == 0
    assert result['A'][2] == df['A'].mean()


def test_encode_categorical():
    """Test categorical encoding."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        'cat_col': ['a', 'b', 'a', 'c'],
        'num_col': [1, 2, 3, 4]
    })
    
    result = preprocessor.encode_categorical(df, ['cat_col'])
    
    assert result['cat_col'].dtype in [np.int32, np.int64]
    assert len(result['cat_col'].unique()) == 3


def test_scale_features():
    """Test feature scaling."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40]
    })
    
    result = preprocessor.scale_features(df, ['A', 'B'])
    
    # Check that scaled values have mean close to 0 and std close to 1
    assert abs(result['A'].mean()) < 0.1
    assert abs(result['A'].std() - 1.0) < 0.1
