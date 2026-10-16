"""
Streamlit dashboard for the term deposit model, built on the held-out test split only.

Every score comes from the prediction API (API_URL); the dashboard never loads the
model. The test split is the newest 15% of customers, written to
data/processed/test.parquet by the training pipeline.

    docker compose up ui      # also starts the API; open http://localhost:8501
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from src.ui.analysis import (age_band, call_top, gains_by_decile, position_among, rank, roc_auc,
                             segment_rates, time_periods)

API_URL = os.getenv('API_URL', 'http://localhost:8080')
TEST_DATA = Path(os.getenv('TEST_DATA', 'data/processed/test.parquet'))
NUMERIC = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


@st.cache_data
def load_test() -> pd.DataFrame:
    return pd.read_parquet(TEST_DATA)


@st.cache_data(ttl=60)
def model_info() -> dict:
    response = httpx.get(f'{API_URL}/model', timeout=10)
    response.raise_for_status()
    return response.json()


def predict(records: list) -> list:
    response = httpx.post(f'{API_URL}/predict', json={'instances': records}, timeout=120)
    response.raise_for_status()
    return response.json()['predictions']


@st.cache_data
def score_test(run_id: str) -> pd.DataFrame:
    """Score every test customer once per served model (the run id keys the cache)."""
    test = load_test()
    predictions = predict(test.drop(columns='y').to_dict(orient='records'))
    return test.assign(probability=[p['probability'] for p in predictions])


st.set_page_config(page_title='Term deposit model', layout='wide')
st.title('Term deposit model: who to call')

if not TEST_DATA.exists():
    st.error(f'No test split at {TEST_DATA}. Run training first: docker compose run --rm train')
    st.stop()
try:
    model = model_info()
    scored = score_test(model['run_id'])
except httpx.HTTPError as exc:
    st.error(f'The prediction API at {API_URL} is not answering ({exc}). Start it: docker compose up api')
    st.stop()

test = load_test()
st.caption(f"Model: {model['model_type']} · trained {model['trained_at'][:16]} · commit {model['commit'][:7]} · "
           f"cut-off {model['decision_threshold']:.3f} · all figures use the held-out test split "
           f"({len(test):,} newest customers, never used for training)")

predict_tab, call_tab, data_tab, model_tab = st.tabs(
    ['Predict a customer', 'Who to call', 'Test data analytics', 'Model performance'])

with predict_tab:
    st.subheader('Should we call this customer?')
    with st.form('customer'):
        columns = st.columns(3)
        customer = {}
        for i, name in enumerate(NUMERIC):
            customer[name] = columns[i % 3].number_input(name, value=int(test[name].median()), step=1)
        for i, name in enumerate(CATEGORICAL):
            options = sorted(test[name].unique())
            customer[name] = columns[i % 3].selectbox(name, options, index=options.index(test[name].mode()[0]))
        submitted = st.form_submit_button('Predict')
    if submitted:
        try:
            result = predict([customer])[0]
        except httpx.HTTPError as exc:
            st.error(f'Prediction failed: {exc}')
        else:
            left, middle, right = st.columns(3)
            left.metric('Probability of subscribing', f"{result['probability']:.1%}")
            middle.metric('Decision at the model cut-off', 'Call' if result['subscribe'] else "Don't call")
            right.metric('Rank among test customers', f"top {position_among(result['probability'], scored):.0f}%")

with call_tab:
    st.subheader('If we can only call some customers, who and what do we gain?')
    top_percent = st.slider('Call the top % of the ranked list', 1, 100, 10)
    summary = call_top(scored, top_percent)
    cols = st.columns(4)
    cols[0].metric('Calls', f"{summary['calls']:,}")
    cols[1].metric('Subscribers reached', f"{summary['subscribers']:,}",
                   f"{summary['subscribers'] - summary['random_subscribers']:+,.0f} vs random calling")
    cols[2].metric('Conversion', f"{summary['conversion']:.1%}", f"lift {summary['lift']:.2f}x")
    cols[3].metric('Share of all subscribers found', f"{summary['share_of_all_subscribers']:.0%}")
    called = rank(scored).head(summary['calls'])
    st.dataframe(called.drop(columns='y'), hide_index=True, height=320)
    st.download_button('Download the call list (CSV)', called.drop(columns='y').to_csv(index=False),
                       file_name=f'call_list_top_{top_percent}pct.csv', mime='text/csv')

with data_tab:
    st.subheader('What do the test-period customers look like?')
    cols = st.columns(3)
    cols[0].metric('Customers', f'{len(test):,}')
    cols[1].metric('Subscribed', f"{test['y'].mean():.1%}")
    cols[2].metric('Contact months covered', time_periods(test['month']).nunique())
    st.markdown('**Subscriber rate by contact month** (in date order)')
    by_month = test['y'].groupby(time_periods(test['month']), sort=False).mean()
    st.line_chart(by_month.rename('subscriber rate'))
    segments = {'Previous campaign outcome': test['poutcome'], 'Housing loan': test['housing'],
                'Contact type': test['contact'], 'Age band': age_band(test['age']), 'Job': test['job']}
    left, right = st.columns(2)
    for i, (title, column) in enumerate(segments.items()):
        with (left if i % 2 == 0 else right):
            st.markdown(f'**Subscriber rate by {title.lower()}**')
            rates = segment_rates(test, column).reset_index()
            st.bar_chart(rates, x=rates.columns[0], y='subscriber_rate', horizontal=True, sort='-subscriber_rate')
    st.markdown("**Share of 'unknown' values**")
    unknown = {name: (test[name] == 'unknown').mean() for name in CATEGORICAL if (test[name] == 'unknown').any()}
    st.dataframe(pd.Series(unknown, name='unknown share').map('{:.1%}'.format))

with model_tab:
    st.subheader('How good is the model on customers it never saw?')
    gains = gains_by_decile(scored)
    cols = st.columns(4)
    cols[0].metric('ROC-AUC', f"{roc_auc(scored['y'], scored['probability']):.3f}")
    cols[1].metric('Lift in the top 10%', f"{gains['lift'].iloc[0]:.2f}x",
                   f"best possible {1 / scored['y'].mean():.2f}x", delta_color='off')
    cols[2].metric('Subscribers in the top 10%', f"{gains['cumulative_share_of_subscribers'].iloc[0]:.0%}")
    cols[3].metric('Subscribers in the top 30%', f"{gains['cumulative_share_of_subscribers'].iloc[2]:.0%}")
    left, right = st.columns(2)
    with left:
        st.markdown('**Gains: share of subscribers found vs share of customers called**')
        curve = pd.DataFrame({'model': [0.0, *gains['cumulative_share_of_subscribers']],
                              'random calling': [d / 10 for d in range(11)]},
                             index=pd.Index(range(0, 101, 10), name='% of customers called'))
        st.line_chart(curve)
    with right:
        st.markdown('**Lift by tenth of the ranked list** (1.0 = random calling)')
        st.bar_chart(gains['lift'])
    st.markdown('**Recorded at training** (same test split)')
    st.dataframe(pd.Series(model['test_metrics'], name='value').round(3))
    with st.expander('Model lineage'):
        st.json(model)
