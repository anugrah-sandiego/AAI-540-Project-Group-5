"""Tests for the dashboard calculations (src/ui/analysis.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.ui.analysis import age_band, call_top, gains_by_decile, roc_auc, segment_rates, time_periods


def scored_customers(n=100, seed=0):
    """Customers whose probability loosely follows the outcome, 30% subscribers."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame({'y': y, 'probability': np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)})


def test_roc_auc_matches_scikit_learn():
    scored = scored_customers()
    expected = roc_auc_score(scored['y'], scored['probability'])
    assert roc_auc(scored['y'], scored['probability']) == pytest.approx(expected)


def test_calling_the_top_of_a_perfect_ranking_finds_only_subscribers():
    scored = pd.DataFrame({'y': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'probability': np.linspace(1, 0.1, 10)})
    summary = call_top(scored, top_percent=20)
    assert summary['calls'] == 2 and summary['subscribers'] == 2
    assert summary['conversion'] == 1.0 and summary['lift'] == pytest.approx(5.0)
    assert summary['random_subscribers'] == pytest.approx(0.4)
    assert summary['share_of_all_subscribers'] == 1.0


def test_gains_cover_every_customer_and_end_with_all_subscribers():
    scored = scored_customers()
    gains = gains_by_decile(scored)
    assert list(gains.index) == list(range(1, 11)) and gains['customers'].sum() == 100
    assert gains['cumulative_share_of_subscribers'].iloc[-1] == pytest.approx(1.0)
    assert gains['lift'].iloc[0] >= gains['lift'].iloc[-1]


def test_time_periods_keep_repeated_months_apart():
    months = pd.Series(['nov', 'nov', 'dec', 'jan', 'nov'])
    assert time_periods(months).tolist() == ['01 nov', '01 nov', '02 dec', '03 jan', '04 nov']


def test_segment_rates_and_age_bands():
    test = pd.DataFrame({'y': [1, 0, 1, 1], 'age': [25, 35, 61, 64]})
    bands = age_band(test['age'])
    assert bands.tolist() == ['under 30', '30-39', '60 and over', '60 and over']
    rates = segment_rates(test, bands)
    assert rates.loc['60 and over', 'customers'] == 2 and rates['subscriber_rate'].iloc[0] == 1.0
