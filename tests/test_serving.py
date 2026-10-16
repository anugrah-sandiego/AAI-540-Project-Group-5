"""Tests for the prediction service (SageMaker /ping and /invocations, local /predict)."""

import pytest
from fastapi.testclient import TestClient

from src.serving import app as serving
from tests.model_fixtures import save_model_dir
from tests.test_data_preprocessing import make_raw_df


@pytest.fixture
def model_dir(tmp_path):
    return save_model_dir(tmp_path / 'model')


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv('MODEL_DIR', str(model_dir))
    serving.get_predictor.cache_clear()
    yield TestClient(serving.app)
    serving.get_predictor.cache_clear()


def records(n=2):
    return make_raw_df(n=n, seed=5).drop(columns=['y', 'duration']).to_dict(orient='records')


def test_ping_is_healthy_once_the_model_loads(client):
    assert client.get('/ping').status_code == 200


def test_ping_reports_unhealthy_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'missing'))
    serving.get_predictor.cache_clear()
    assert TestClient(serving.app).get('/ping').status_code == 503
    serving.get_predictor.cache_clear()


@pytest.mark.parametrize('path', ['/invocations', '/predict'])
def test_predictions_carry_probability_decision_and_model_version(client, path):
    response = client.post(path, json={'instances': records(3)})
    assert response.status_code == 200
    body = response.json()
    assert body['model'] == {'run_id': 'run-0', 'commit': 'abc1234',
                             'trained_at': '2026-10-15T00:00:00+00:00', 'decision_threshold': 0.5}
    assert len(body['predictions']) == 3
    for prediction in body['predictions']:
        assert 0 <= prediction['probability'] <= 1
        assert prediction['subscribe'] == (prediction['probability'] >= 0.5)


def test_model_endpoint_describes_the_served_model(client):
    body = client.get('/model').json()
    assert body['run_id'] == 'run-0' and body['commit'] == 'abc1234' and body['decision_threshold'] == 0.5
    assert {'model_type', 'refit_on_all_data', 'test_metrics', 'data_sha256'} <= set(body)


def test_missing_columns_are_rejected_by_name(client):
    record = records(1)[0]
    del record['age'], record['job']
    response = client.post('/predict', json={'instances': [record]})
    assert response.status_code == 422
    assert "['age', 'job']" in response.json()['detail']


def test_non_numeric_value_in_numeric_column_is_rejected(client):
    record = records(1)[0] | {'balance': 'lots'}
    assert client.post('/predict', json={'instances': [record]}).status_code == 422


def test_empty_request_is_rejected(client):
    assert client.post('/predict', json={'instances': []}).status_code == 422
