"""
Calculations behind the dashboard (src/ui/app.py), kept free of Streamlit so they
can be tested. They use pandas only: the dashboard gets every score from the
prediction API and never loads the model or scikit-learn.

`scored` frames hold one row per test customer with the true outcome `y` (0/1)
and the API's `probability`.
"""

from typing import Dict

import pandas as pd

AGE_BANDS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ['under 30', '30-39', '40-49', '50-59', '60 and over']


def roc_auc(y: pd.Series, score: pd.Series) -> float:
    """Chance that a random subscriber is scored above a random non-subscriber (ties count half)."""
    ranks = score.rank()
    positives = int(y.sum())
    negatives = len(y) - positives
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def rank(scored: pd.DataFrame) -> pd.DataFrame:
    """Highest probability first, with each customer's position as a share of the list."""
    ranked = scored.sort_values('probability', ascending=False).reset_index(drop=True)
    ranked['top_percent'] = (ranked.index + 1) / len(ranked) * 100
    return ranked


def call_top(scored: pd.DataFrame, top_percent: float) -> Dict[str, float]:
    """What calling the top share of the ranked list achieves, compared with calling at random."""
    ranked = rank(scored)
    called = ranked.head(max(1, round(len(ranked) * top_percent / 100)))
    base_rate = scored['y'].mean()
    return {'calls': len(called),
            'subscribers': int(called['y'].sum()),
            'conversion': called['y'].mean(),
            'random_subscribers': base_rate * len(called),
            'lift': called['y'].mean() / base_rate,
            'share_of_all_subscribers': called['y'].sum() / scored['y'].sum()}


def decile(ranked: pd.DataFrame) -> pd.Series:
    """Tenth of a ranked list each customer falls in: 1 = highest scores."""
    return pd.Series(ranked.index * 10 // len(ranked) + 1, index=ranked.index, name='decile')


def gains_by_decile(scored: pd.DataFrame) -> pd.DataFrame:
    """Conversion and lift per tenth of the ranked list, and the share of subscribers found so far."""
    ranked = rank(scored)
    table = ranked['y'].groupby(decile(ranked)).agg(customers='size', subscribers='sum', conversion='mean')
    table['lift'] = table['conversion'] / scored['y'].mean()
    table['cumulative_share_of_subscribers'] = table['subscribers'].cumsum() / scored['y'].sum()
    return table


def actual_vs_predicted(scored: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Actual subscriber rate next to the average predicted probability, per group (in group order)."""
    return scored.groupby(group, sort=False).agg(actual=('y', 'mean'), predicted=('probability', 'mean'))


def time_periods(month: pd.Series) -> pd.Series:
    """Label each contact month in file order (rows are in date order; the data has no year column)."""
    period = (month != month.shift()).cumsum()
    return period.map(lambda p: f'{p:02d}') + ' ' + month


def segment_rates(test: pd.DataFrame, column: pd.Series) -> pd.DataFrame:
    """Customers and subscriber rate per value of a column, highest rate first."""
    table = test['y'].groupby(column).agg(customers='size', subscriber_rate='mean')
    return table.sort_values('subscriber_rate', ascending=False)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS, right=False)
