"""A small trained model saved the way the training pipeline saves it (used by serving and step tests)."""

import json

import joblib

from src.data_ingestion.data_ingestion import CATEGORICAL_COLUMNS
from src.data_preprocessing.data_preprocessing import encode_target
from src.feature_engineering.feature_engineering import DOMAIN_FEATURES
from src.models.train_model import build_model
from src.utils.lineage import input_schema
from tests.test_data_preprocessing import make_raw_df

NUMERIC = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous'] + DOMAIN_FEATURES


def save_model_dir(path, seed=0, threshold=0.5):
    """Train a logistic regression on synthetic data and write model.joblib + metadata."""
    df = make_raw_df(n=300, seed=seed)
    X, y = df.drop(columns=['y', 'duration']), encode_target(df['y'])
    model = build_model('logistic_regression', NUMERIC, CATEGORICAL_COLUMNS, 'none').fit(X, y)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / 'model.joblib')
    metadata = {'run_id': f'run-{seed}', 'decision_threshold': threshold, 'trained_at': '2026-10-15T00:00:00+00:00',
                'lineage': {'git': {'commit': 'abc1234'}}, 'input_schema': input_schema(X)}
    (path / 'model_metadata.json').write_text(json.dumps(metadata))
    return path
