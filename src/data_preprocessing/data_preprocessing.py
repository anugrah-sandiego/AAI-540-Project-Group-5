"""
Data preprocessing module for cleaning and preparing data.

All transformations are expressed as scikit-learn transformers so they are fitted
on the training split only and persisted together with the model.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List


def encode_target(y: pd.Series) -> pd.Series:
    """
    Map the yes/no target to 1/0.

    Args:
        y: Target Series with 'yes'/'no' values

    Returns:
        Integer target Series
    """
    return y.map({'yes': 1, 'no': 0}).astype(int)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build the column transformer used by every model.

    Nominal categoricals are one-hot encoded (label encoding would impose a false
    ordering on columns such as job or month). Categories unseen during training
    are ignored at inference time instead of raising an error.

    Args:
        numeric_cols: List of numerical column names
        categorical_cols: List of categorical column names

    Returns:
        Unfitted ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False,
    )
