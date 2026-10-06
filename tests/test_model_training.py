"""
Tests for model training and evaluation modules.
"""

import pytest
import numpy as np
from src.data_ingestion.data_ingestion import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.data_preprocessing.data_preprocessing import encode_target
from src.feature_engineering.feature_engineering import DOMAIN_FEATURES
from src.models.train_model import build_model
from src.models.predict_model import find_best_threshold, lift_at_k, evaluate_model, predict_model
from tests.test_data_preprocessing import make_raw_df


NUMERIC = [c for c in NUMERIC_COLUMNS if c != 'duration'] + DOMAIN_FEATURES


@pytest.mark.parametrize('model_type', ['logistic_regression', 'decision_tree',
                                        'random_forest', 'gradient_boosting'])
@pytest.mark.parametrize('strategy', ['none', 'class_weight', 'smote'])
def test_build_model_fits_and_predicts_on_raw_columns(model_type, strategy):
    df = make_raw_df(n=300)
    X, y = df.drop(columns=['y', 'duration']), encode_target(df['y'])
    model = build_model(model_type, NUMERIC, CATEGORICAL_COLUMNS, strategy)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (300,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_class_weight_strategy_sets_balanced_weights():
    model = build_model('random_forest', NUMERIC, CATEGORICAL_COLUMNS, 'class_weight')
    assert model.named_steps['classifier'].class_weight == 'balanced'
    assert 'smote' not in model.named_steps


def test_smote_strategy_adds_resampling_step():
    model = build_model('logistic_regression', NUMERIC, CATEGORICAL_COLUMNS, 'smote')
    assert 'smote' in model.named_steps
    assert model.named_steps['classifier'].class_weight is None


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_model('random_forest', NUMERIC, CATEGORICAL_COLUMNS, 'undersample')


def test_find_best_threshold_separable_data():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = find_best_threshold(y, proba)
    assert result['f1'] == pytest.approx(1.0)
    assert 0.3 < result['threshold'] <= 0.8


def test_lift_at_k():
    # 10 customers, 2 positives, both ranked in the top 20%
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = np.linspace(1, 0, 10)
    result = lift_at_k(y, proba, [10, 20, 50])
    assert result['conversion_at_20'] == pytest.approx(1.0)
    assert result['lift_at_20'] == pytest.approx(5.0)
    assert result['lift_at_50'] == pytest.approx(2.0)


def test_evaluate_model_uses_threshold():
    df = make_raw_df(n=300)
    X, y = df.drop(columns=['y', 'duration']), encode_target(df['y'])
    model = build_model('logistic_regression', NUMERIC, CATEGORICAL_COLUMNS, 'none').fit(X, y)

    low = evaluate_model(model, X, y, threshold=0.0)
    assert low['recall'] == 1.0
    assert {'roc_auc', 'pr_auc', 'lift_at_10', 'conversion_at_30'} <= set(low)
    assert predict_model(model, X, threshold=1.01).sum() == 0
