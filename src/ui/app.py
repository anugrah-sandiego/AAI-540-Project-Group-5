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

from src.ui.analysis import (actual_vs_predicted, age_band, call_top, decile, gains_by_decile, position_among,
                             month_of_year, rank, roc_auc, segment_rates)

API_URL = os.getenv('API_URL', 'http://localhost:8080')
TEST_DATA = Path(os.getenv('TEST_DATA', 'data/processed/test.parquet'))
NUMERIC = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# Inputs a user gives for a potential customer; the model's call-timing inputs
# (day, month, contact, campaign) are filled with typical test values instead
CUSTOMER_DETAILS = {'age': 'number', 'balance': 'number', 'job': 'choice', 'marital': 'choice',
                    'education': 'choice', 'default': 'choice', 'housing': 'choice', 'loan': 'choice'}
HISTORY = {'pdays': 'number', 'previous': 'number', 'poutcome': 'choice'}


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
    # The data is anonymised (no names or phone numbers), so customers are identified
    # by their position in the test split; a bank would join this ID to its CRM
    return test.assign(customer_id=[f'TEST-{i:05d}' for i in range(1, len(test) + 1)],
                       probability=[p['probability'] for p in predictions])


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

call_tab, data_tab, model_tab = st.tabs(['Who to call', 'Test data analytics', 'Model performance'])

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
    call_list = called[['customer_id', 'probability', 'contact', *NUMERIC, *CATEGORICAL[:-3], 'poutcome']].copy()
    call_list.insert(0, 'call_order', range(1, len(call_list) + 1))
    call_list = call_list.rename(columns={'contact': 'channel', 'poutcome': 'previous_outcome'})
    st.markdown(f'**Call list: top {top_percent}% of test customers, in call order** · the data is anonymised, '
                'so customers are identified by ID and contact channel (a bank would add phone numbers from its CRM)')
    st.dataframe(call_list, hide_index=True, height=420,
                 column_config={'probability': st.column_config.ProgressColumn(
                     'probability', format='percent', min_value=0.0, max_value=1.0)})
    st.download_button('Download the call list (CSV)', call_list.to_csv(index=False),
                       file_name=f'call_list_top_{top_percent}pct.csv', mime='text/csv')

    with st.expander('Check a potential customer: customer details and previous-campaign history'):
        assumed = {'day': int(test['day'].median()), 'month': test['month'].mode()[0],
                   'contact': test['contact'].mode()[0], 'campaign': 1}
        st.caption('The model also needs call-timing inputs; these are filled with typical test values: '
                   + ', '.join(f'{name} = {value}' for name, value in assumed.items()) + ' (first call).')
        with st.form('potential_customer'):
            customer = dict(assumed)
            for group, fields in (('Customer details', CUSTOMER_DETAILS), ('Previous-campaign history', HISTORY)):
                st.markdown(f'**{group}**')
                columns = st.columns(4)
                for i, (name, kind) in enumerate(fields.items()):
                    if kind == 'number':
                        customer[name] = columns[i % 4].number_input(name, value=int(test[name].median()), step=1)
                    else:
                        options = sorted(test[name].unique())
                        customer[name] = columns[i % 4].selectbox(
                            name, options, index=options.index(test[name].mode()[0]))
            st.caption('pdays = days since the last contact in a previous campaign (-1 = never contacted)')
            checked = st.form_submit_button('Check')
        if checked:
            try:
                probability = predict([customer])[0]['probability']
            except httpx.HTTPError as exc:
                st.error(f'Prediction failed: {exc}')
            else:
                position = position_among(probability, scored)
                left, middle, right = st.columns(3)
                left.metric('Probability (use for ranking)', f'{probability:.1%}')
                middle.metric('Would rank among test customers', f'top {position:.0f}%')
                decision = 'Call' if position <= top_percent else "Don't call"
                right.metric(f'Call within the top {top_percent}%?', decision)

with data_tab:
    st.subheader('What do the test-period customers look like?')
    cols = st.columns(3)
    cols[0].metric('Customers', f'{len(test):,}')
    cols[1].metric('Subscribed', f"{test['y'].mean():.1%}")
    cols[2].metric('Subscribers', f"{int(test['y'].sum()):,}")
    st.markdown('**Subscriber rate by month of year** (years combined; the data has no year column)')
    by_month = test['y'].groupby(month_of_year(test['month']), observed=True).mean()
    st.bar_chart(by_month.rename('subscriber rate'), sort=False)
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
    st.markdown('**Actual vs predicted**: real subscriber rate next to the average predicted probability')
    st.caption('Close bars mean the probabilities can be read as real chances; actual rates falling from the '
               'first tenth to the last mean the ranking works.')
    left, right = st.columns(2)
    with left:
        st.markdown('**By month of year** (years combined)')
        st.bar_chart(actual_vs_predicted(scored, month_of_year(scored['month'])), stack=False, sort=False)
    with right:
        st.markdown('**By tenth of the ranked list** (1 = highest predictions)')
        ranked = rank(scored)
        st.bar_chart(actual_vs_predicted(ranked, decile(ranked)), stack=False)
    st.markdown('**Recorded at training** (same test split)')
    st.dataframe(pd.Series(model['test_metrics'], name='value').round(3))
    with st.expander('Model lineage'):
        st.json(model)
