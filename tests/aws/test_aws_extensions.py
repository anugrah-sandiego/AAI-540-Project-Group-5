"""
Tests for batch scoring, the Feature Store, the real-time endpoint and its monitoring,
without calling AWS (botocore Stubber). Run in the aws-tools image:
    docker compose run --rm aws-test
"""

import json
from types import SimpleNamespace

import boto3
import pandas as pd
import pytest
from botocore.stub import Stubber

pytest.importorskip('sagemaker')
from src.aws import endpoint, feature_store  # noqa: E402
from src.aws.batch import CHUNK, from_responses, to_requests  # noqa: E402
from src.aws.config import AwsConfig  # noqa: E402
from src.aws.monitor_preprocessor import preprocess_handler  # noqa: E402

ACCOUNT = '1' * 12
CONFIG = AwsConfig(region='us-east-1', account=ACCOUNT, role_arn=f'arn:aws:iam::{ACCOUNT}:role/LabRole',
                   bucket='test-bucket')


def make_raw_df(n):
    """Raw bank rows (the aws-tools image has no scikit-learn, so no shared fixtures)."""
    row = {'age': 41, 'job': 'admin.', 'marital': 'married', 'education': 'secondary', 'default': 'no',
           'balance': 1200, 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'day': 5, 'month': 'may',
           'duration': 180, 'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown', 'y': 'no'}
    return pd.DataFrame([row | {'age': 30 + i} for i in range(n)])


def client(name):
    return boto3.Session(region_name='us-east-1', aws_access_key_id='test',
                         aws_secret_access_key='test').client(name)


def test_batch_requests_chunk_customers_and_responses_keep_their_order():
    customers = make_raw_df(n=CHUNK * 2 + 10).drop(columns=['y', 'duration'])
    lines = to_requests(customers).splitlines()
    assert len(lines) == 3 and len(json.loads(lines[2])['instances']) == 10
    responses = '\n'.join(json.dumps({'predictions': [{'probability': p} for p in pair]})
                          for pair in ([0.1, 0.2], [0.3])) + '\n'
    assert from_responses(responses) == [0.1, 0.2, 0.3]


def test_feature_store_records_keep_pre_call_inputs_with_a_unique_id():
    batch = make_raw_df(n=3)
    rows = feature_store.records(batch, 'b0007', '2026-10-16T10:00:00Z')
    names = [field['FeatureName'] for field in rows[0]]
    assert 'duration' not in names and {'record_id', 'event_time', 'batch_id', 'y'} <= set(names)
    assert [row[0]['ValueAsString'] for row in rows] == ['b0007-00000', 'b0007-00001', 'b0007-00002']
    assert set(names) == {d['FeatureName'] for d in feature_store.DEFINITIONS}


def test_feature_group_is_created_when_missing(monkeypatch):
    monkeypatch.setattr(feature_store.time, 'sleep', lambda seconds: None)
    sagemaker = client('sagemaker')
    with Stubber(sagemaker) as stub:
        stub.add_client_error('describe_feature_group', 'ResourceNotFound')
        stub.add_response('create_feature_group', {'FeatureGroupArn': 'arn:fg'})
        stub.add_response('describe_feature_group', {
            'FeatureGroupArn': 'arn:fg', 'FeatureGroupName': feature_store.FEATURE_GROUP,
            'RecordIdentifierFeatureName': 'record_id', 'EventTimeFeatureName': 'event_time',
            'FeatureDefinitions': feature_store.DEFINITIONS, 'CreationTime': '2026-10-16',
            'FeatureGroupStatus': 'Created', 'NextToken': ''})
        feature_store.ensure_feature_group(CONFIG, sagemaker)
        stub.assert_no_pending_responses()


def test_dashboard_tracks_the_live_endpoint():
    widgets = json.loads(endpoint.dashboard_body('us-east-1'))['widgets']
    assert len(widgets) == len(endpoint.METRICS)
    assert all(endpoint.ENDPOINT_NAME in widget['properties']['metrics'][0] for widget in widgets)


def test_endpoint_down_removes_everything_even_when_parts_are_gone():
    sagemaker, cloudwatch = client('sagemaker'), client('cloudwatch')
    aws = SimpleNamespace(client=lambda name: {'sagemaker': sagemaker, 'cloudwatch': cloudwatch}[name])
    with Stubber(sagemaker) as stub, Stubber(cloudwatch) as cw:
        stub.add_client_error('delete_monitoring_schedule', 'ResourceNotFound')
        stub.add_client_error('describe_monitoring_schedule', 'ResourceNotFound')
        stub.add_response('delete_endpoint', {})
        stub.add_response('delete_endpoint_config', {})
        stub.add_response('delete_model', {})
        cw.add_response('delete_dashboards', {})
        endpoint.endpoint_down(aws)
        stub.assert_no_pending_responses()
        cw.assert_no_pending_responses()


def test_monitor_preprocessor_checks_the_first_customer_of_a_request():
    customer = make_raw_df(n=2).drop(columns=['y', 'duration']).to_dict(orient='records')
    record = SimpleNamespace(endpoint_input=SimpleNamespace(data=json.dumps({'instances': customer})))
    assert preprocess_handler(record) == json.loads(json.dumps(customer[0]))


def test_monitoring_baseline_uses_model_inputs_only():
    assert list(endpoint.model_inputs(make_raw_df(n=2)).columns) == [
        c for c in make_raw_df(n=2).columns if c not in ('y', 'duration')]


def test_prepare_step_writes_to_the_feature_store_in_the_pipeline_region():
    from sagemaker.workflow.pipeline_context import PipelineSession
    from src.aws.pipeline import build_pipeline
    fake = boto3.Session(region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
    session = PipelineSession(boto_session=fake, default_bucket='test-bucket')
    steps = json.loads(build_pipeline(CONFIG, 'abc1234', session).definition())['Steps']
    prepare = next(step for step in steps if step['Name'] == 'PrepareData')
    assert prepare['Arguments']['AppSpecification']['ContainerArguments'] == ['--region', 'us-east-1']
