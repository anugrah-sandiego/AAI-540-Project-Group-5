"""
Data preprocessing module for cleaning and preparing data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple


class DataPreprocessor:
    """Handle data preprocessing operations."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col])
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        return df
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numerical column names
            fit: Whether to fit the scaler or use existing
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        for col in numeric_cols:
            if col in df.columns:
                if col not in self.scalers or fit:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    df[col] = self.scalers[col].transform(df[[col]])
        
        return df
