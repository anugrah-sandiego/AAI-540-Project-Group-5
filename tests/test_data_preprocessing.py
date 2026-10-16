"""
Tests for data ingestion, preprocessing and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_ingestion.data_ingestion import validate_schema, split_data
from src.data_preprocessing.data_preprocessing import encode_target, build_preprocessor
from src.feature_engineering.feature_engineering import add_domain_features, DOMAIN_FEATURES


def make_raw_df(n=200, seed=0):
    """Small synthetic frame with the bank-full.csv schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'age': rng.integers(18, 90, n),
        'job': rng.choice(['admin.', 'technician', 'unknown'], n),
        'marital': rng.choice(['married', 'single'], n),
        'education': rng.choice(['primary', 'secondary', 'tertiary'], n),
        'default': rng.choice(['yes', 'no'], n),
        'balance': rng.integers(-500, 5000, n),
        'housing': rng.choice(['yes', 'no'], n),
        'loan': rng.choice(['yes', 'no'], n),
        'contact': rng.choice(['cellular', 'unknown'], n),
        'day': rng.integers(1, 31, n),
        'month': rng.choice(['may', 'jun', 'jul'], n),
        'duration': rng.integers(0, 1000, n),
        'campaign': rng.integers(1, 10, n),
        'pdays': rng.choice([-1, 100, 200], n),
        'previous': rng.integers(0, 5, n),
        'poutcome': rng.choice(['unknown', 'success', 'failure'], n),
        'y': rng.choice(['yes', 'no'], n, p=[0.2, 0.8]),
    })


def test_validate_schema_passes_on_valid_data():
    report = validate_schema(make_raw_df())
    assert report['n_rows'] == 200
    assert 0 < report['positive_rate'] < 1
    assert 'poutcome' in report['unknown_rates']


def test_validate_schema_rejects_missing_column():
    with pytest.raises(ValueError, match='Missing required columns'):
        validate_schema(make_raw_df().drop(columns=['balance']))


def test_validate_schema_rejects_bad_target():
    df = make_raw_df()
    df.loc[0, 'y'] = 'maybe'
    with pytest.raises(ValueError, match='Unexpected target values'):
        validate_schema(df)


def test_split_is_70_15_15_stratified_and_disjoint():
    df = make_raw_df(n=1000)
    X, y = df.drop(columns='y'), encode_target(df['y'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.15, 0.15, 42)

    assert len(X_train) == 700 and len(X_val) == 150 and len(X_test) == 150
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    for part in (y_train, y_val, y_test):
        assert abs(part.mean() - y.mean()) < 0.02


def test_encode_target():
    assert encode_target(pd.Series(['yes', 'no', 'no'])).tolist() == [1, 0, 0]


def test_domain_features_are_rowwise():
    df = make_raw_df()
    result = add_domain_features(df)
    assert set(DOMAIN_FEATURES) <= set(result.columns)
    assert (result['previously_contacted'] == (df['pdays'] != -1)).all()
    assert (result['prior_success'] == (df['poutcome'] == 'success')).all()
    # Row-wise: features for a subset equal the subset of features
    assert result.iloc[:10].equals(add_domain_features(df.iloc[:10]))


def test_preprocessor_one_hot_encodes_and_ignores_unseen_categories():
    df = make_raw_df()
    pre = build_preprocessor(['age', 'balance'], ['job', 'month'])
    pre.fit(df)

    unseen = df.head(2).copy()
    unseen['job'] = 'astronaut'
    out = pre.transform(unseen)
    job_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith('job_')]
    # Unknown category -> all-zero one-hot block instead of an error
    assert out[:, job_cols].sum() == 0
    assert len(job_cols) == df['job'].nunique()
