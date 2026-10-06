"""
Feature engineering module for creating and selecting features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Tuple


class FeatureEngineer:
    """Handle feature engineering operations."""
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                    feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between pairs of columns.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing column pairs to interact
            
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   cols: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df: Input DataFrame
            cols: List of column names
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()
        
        for col in cols:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f"{col}_poly_{d}"] = df[col] ** d
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'k_best', k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('k_best', 'mutual_info')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        if method == 'k_best':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
