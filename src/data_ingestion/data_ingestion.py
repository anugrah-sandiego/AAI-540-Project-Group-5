"""
Data ingestion module for loading, validating and splitting raw data.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Tuple
from sklearn.model_selection import train_test_split


# Expected schema of the UCI Bank Marketing dataset (bank-full.csv)
NUMERIC_COLUMNS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL_COLUMNS = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'poutcome']
TARGET_COLUMN = 'y'
EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]


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
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
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


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Run schema and data-quality checks on the raw bank marketing data.

    Checks column presence, numeric types, nulls, target values and duplicates.
    'unknown' values are reported but not treated as errors, because they are a
    legitimate category in this dataset.

    Args:
        df: Raw DataFrame

    Returns:
        Dictionary summarising the data-quality report
    """
    validate_data(df, EXPECTED_COLUMNS)

    non_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns have non-numeric types: {non_numeric}")

    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    invalid_target = set(df[TARGET_COLUMN].unique()) - {'yes', 'no'}
    if invalid_target:
        raise ValueError(f"Unexpected target values: {invalid_target}")

    unknown_rates = {c: round(float((df[c] == 'unknown').mean()), 4)
                     for c in CATEGORICAL_COLUMNS if (df[c] == 'unknown').any()}

    return {
        'n_rows': len(df),
        'n_duplicates': int(df.duplicated().sum()),
        'positive_rate': round(float((df[TARGET_COLUMN] == 'yes').mean()), 4),
        'unknown_rates': unknown_rates,
    }


def split_data(X: pd.DataFrame, y: pd.Series, val_size: float = 0.15,
               test_size: float = 0.15, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                          pd.Series, pd.Series, pd.Series]:
    """
    Stratified train / validation / test split.

    The source file is ordered chronologically and the positive rate rises
    sharply towards the end, so rows are shuffled before splitting.

    Args:
        X: Feature DataFrame
        y: Target Series
        val_size: Fraction of all rows used for validation
        test_size: Fraction of all rows used for the final test
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Rescale val_size so it is a fraction of the full dataset
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction,
        stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
