"""Failure gates, AWS boundary, and serving regression tests; no AWS credentials."""
import io
import json
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from cicd import release
from cicd.container import inference
from cicd.container.train import build_pipeline, separate


@pytest.mark.parametrize('metrics', [
    {'roc_auc': .74, 'lift_at_10': 3},
    {'roc_auc': .8, 'lift_at_10': 2.49},
    {'roc_auc': float('nan'), 'lift_at_10': 3},
])
def test_bad_candidate_blocks_release(metrics):
    with pytest.raises(ValueError, match='gate failed'):
        release.gates(metrics)


def test_gate_accepts_boundary():
    release.gates({'roc_auc': .75, 'lift_at_10': 2.5})


def test_unapproved_package_cannot_touch_production(monkeypatch):
    monkeypatch.setattr(release, 'load', lambda: {'staging_passed': True, 'package_arn': 'candidate'})
    sm = Mock()
    sm.describe_model_package.return_value = {'ModelApprovalStatus': 'PendingManualApproval'}
    monkeypatch.setattr(release, 'client', lambda service: sm)
    with pytest.raises(ValueError, match='not approved'):
        release.promote()
    sm.create_model.assert_not_called()
    sm.update_endpoint.assert_not_called()


def test_access_denied_is_not_missing():
    err = ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'Denied'}}, 'DescribeEndpoint')
    assert not release.missing(err)


def test_s3_get_uses_exact_bucket_and_key(monkeypatch):
    s3 = Mock(); s3.get_object.return_value = {'Body': io.BytesIO(b'data')}
    monkeypatch.setattr(release, 'client', lambda service: s3)
    assert release.read_s3('s3://my-bucket/processed/train.parquet') == b'data'
    s3.get_object.assert_called_once_with(Bucket='my-bucket', Key='processed/train.parquet')


def test_persisted_preprocessing_supports_unseen_category_and_order(tmp_path):
    import joblib
    x = pd.DataFrame({'age': list(range(40)), 'job': ['admin', 'tech'] * 20})
    model, numeric, categorical = build_pipeline(x, trees=5)
    model.fit(x, np.array([0, 1] * 20))
    bundle = dict(pipeline=model, numeric=numeric, categorical=categorical,
                  columns=x.columns.tolist(), threshold=.5)
    joblib.dump(bundle, tmp_path / 'model.joblib')
    loaded = inference.model_fn(tmp_path)
    sample = pd.DataFrame({'job': ['astronaut'], 'age': [25]})
    result = inference.predict_fn(sample.copy(), loaded)
    assert len(result['probabilities']) == 1
    assert 0 <= result['probabilities'][0] <= 1
    assert result == inference.predict_fn(sample[['age', 'job']].copy(), loaded)


def test_target_and_leakage_removed():
    x, y = separate(pd.DataFrame({'age': [20, 30], 'duration': [999, 888], 'y': ['no', 'yes']}))
    assert list(x.columns) == ['age'] and y.tolist() == [0, 1]


def test_serving_rejects_missing_features():
    with pytest.raises(ValueError, match='schema'):
        inference.predict_fn(pd.DataFrame({'wrong': [1]}), {'columns': ['age']})


def test_smoke_mismatch_blocks_promotion(monkeypatch):
    runtime = Mock()
    runtime.invoke_endpoint.return_value = {'Body': io.BytesIO(json.dumps({
        'probabilities': [.9], 'predictions': [1]}).encode())}
    monkeypatch.setattr(release, 'client', lambda service: runtime)
    fixture = {'request': {'instances': [{'age': 20}]},
               'expected': {'probabilities': [.2], 'predictions': [0]}}
    with pytest.raises(ValueError, match='mismatch'):
        release.smoke('staging', fixture)


def test_failed_staging_never_promotes(monkeypatch):
    monkeypatch.setattr(release, 'load', lambda: {'staging_passed': False})
    aws = Mock(); monkeypatch.setattr(release, 'client', aws)
    with pytest.raises(ValueError, match='Staging must pass'):
        release.promote()
    aws.assert_not_called()


def test_post_update_failure_restores_exact_previous_config(monkeypatch):
    state = {'staging_passed': True, 'package_arn': 'candidate', 'smoke': {}}
    sm = Mock(); sm.describe_model_package.return_value = {'ModelApprovalStatus': 'Approved'}
    monkeypatch.setattr(release, 'load', lambda: state)
    monkeypatch.setattr(release, 'save', lambda s: None)
    monkeypatch.setattr(release, 'client', lambda service: sm)
    infos = iter([{'EndpointStatus': 'InService', 'EndpointConfigName': 'old-exact-config'},
                  {'EndpointStatus': 'InService', 'EndpointConfigName': 'new-config'}])
    monkeypatch.setattr(release, 'endpoint_info', lambda name: next(infos))
    monkeypatch.setattr(release, 'config', lambda s, kind: 'new-config')
    monkeypatch.setattr(release, 'monitoring', lambda name: 'alarm')
    monkeypatch.setattr(release, 'wait', Mock(side_effect=[RuntimeError('bad deployment'), None]))
    with pytest.raises(RuntimeError, match='bad deployment'):
        release.promote()
    assert sm.update_endpoint.call_count == 2
    assert sm.update_endpoint.call_args.kwargs['EndpointConfigName'] == 'old-exact-config'
    assert state['rollback_status'] == 'restored_previous_config'
