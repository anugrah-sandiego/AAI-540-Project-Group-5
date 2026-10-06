"""
Data ingestion module for loading and validating raw data.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_data(file_path: Union[str, Path], sep: str = ',') -> pd.DataFrame:
    """
    Load data from a file (CSV, Excel, or other formats).
    
    Args:
        file_path: Path to the data file
        sep: Separator for CSV files (default: ',')
        
    Returns:
        DataFrame containing the loaded data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path, sep=sep)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that the DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
