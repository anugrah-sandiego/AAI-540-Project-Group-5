"""CI entrypoints and Feature Store retries without AWS access."""
from datetime import datetime, timezone
import io
import json
import runpy
from unittest.mock import Mock
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from src.aws.sagemaker import ci_entrypoints as bridge
from src.aws.sagemaker.data_upload import S3DataUploader
from tests.test_data_preprocessing import make_raw_df


@pytest.mark.parametrize('name', ['training', 'feature_store', 'model_registry', 'deployment'])
def test_original_cli_routes_before_sdk_import(monkeypatch, name):
    monkeypatch.setenv('GROUP5_CI', '1')
    dispatch = Mock(); monkeypatch.setattr(bridge, 'run', dispatch)
    with pytest.raises(SystemExit) as result:
        runpy.run_module('src.aws.sagemaker.' + name, run_name='__main__')
    assert result.value.code == 0
    dispatch.assert_called_once_with(name)


def test_training_exports_actual_artifact_and_prepares_splits(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bridge.release, 'prepare', lambda: calls.append('prepare'))
    monkeypatch.setattr(bridge.release, 'train', lambda: calls.append('train'))
    monkeypatch.setattr(bridge.release, 'load', lambda: {
        'training_job': 'new-job', 'model_uri': 's3://bucket/new/output/model.tar.gz'})
    path = tmp_path / 'env'; monkeypatch.setenv('GITHUB_ENV', str(path))
    bridge.run('training')
    assert calls == ['prepare', 'train']
    assert 'TRAINING_JOB_NAME=new-job\n' in path.read_text()
    assert 'MODEL_ARTIFACT_URI=s3://bucket/new/output/model.tar.gz\n' in path.read_text()


def test_failed_staging_blocks_promotion(monkeypatch):
    monkeypatch.setattr(bridge.release, 'approve', Mock())
    monkeypatch.setattr(bridge.release, 'staging', Mock(side_effect=ValueError('bad parity')))
    promote = Mock(); monkeypatch.setattr(bridge.release, 'promote', promote)
    with pytest.raises(ValueError, match='parity'):
        bridge.run('deployment')
    promote.assert_not_called()


def setup_feature_store(monkeypatch):
    frame = make_raw_df(12)
    raw = frame.to_csv(index=False, sep=';').encode()
    schema_frame = frame.copy()
    schema_frame['customer_id'] = ['id'] * len(frame)
    schema_frame['event_time'] = 1.0
    group = {'FeatureGroupStatus': 'Created', 'RecordIdentifierFeatureName': 'customer_id',
             'EventTimeFeatureName': 'event_time', 'FeatureDefinitions': bridge.feature_definitions(schema_frame),
             'FeatureGroupArn': 'arn:aws:sagemaker:us-east-1:123456789012:feature-group/bank-marketing-features',
             'CreationTime': datetime(2026, 10, 1, tzinfo=timezone.utc),
             'OfflineStoreConfig': {'S3StorageConfig': {'S3Uri': 's3://bucket/features/'}}}
    sm = Mock(); sm.describe_feature_group.return_value = group
    s3 = Mock(); runtime = Mock()
    monkeypatch.setattr(bridge.release, 'read_s3', lambda uri: raw)
    monkeypatch.setattr(bridge.release, 'client', lambda service: {'sagemaker': sm, 's3': s3}[service])
    monkeypatch.setattr(bridge.boto3, 'client', lambda *a, **k: runtime)
    for name, value in {'AWS_REGION': 'us-east-1', 'AWS_BUCKET': 'bucket',
                        'AWS_ROLE_ARN': 'role', 'RAW_DATA_URI': 's3://bucket/raw.csv'}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('FORCE_FEATURE_INGESTION', raising=False)
    monkeypatch.setattr(bridge, 'write_feature_state', Mock())
    return raw, group, sm, s3, runtime


def test_completed_unchanged_ingestion_is_skipped(monkeypatch):
    import hashlib
    raw, group, sm, s3, runtime = setup_feature_store(monkeypatch)
    s3.get_object.return_value = {'Body': io.BytesIO(json.dumps({
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'rows': 12}).encode())}
    bridge.sync_feature_store()
    runtime.put_record.assert_not_called()
    sm.create_feature_group.assert_not_called()


def test_ingestion_writes_checkpoint_only_after_all_rows(monkeypatch):
    raw, group, sm, s3, runtime = setup_feature_store(monkeypatch)
    s3.get_object.side_effect = ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    bridge.sync_feature_store()
    assert runtime.put_record.call_count == 12
    assert runtime.put_record.call_args.kwargs['TargetStores'] == ['OfflineStore']
    s3.put_object.assert_called_once()
    payload = json.loads(s3.put_object.call_args.kwargs['Body'])
    assert payload['rows'] == 12 and payload['status'] == 'ingested'


def test_failed_ingestion_never_marks_completed(monkeypatch):
    raw, group, sm, s3, runtime = setup_feature_store(monkeypatch)
    s3.get_object.side_effect = ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    runtime.put_record.side_effect = RuntimeError('denied')
    with pytest.raises(RuntimeError, match='denied'):
        bridge.sync_feature_store()
    s3.put_object.assert_not_called()


def test_schema_mismatch_is_explicit(monkeypatch):
    raw, group, sm, s3, runtime = setup_feature_store(monkeypatch)
    group['FeatureDefinitions'] = []
    with pytest.raises(ValueError, match='schema differs'):
        bridge.sync_feature_store()
    runtime.put_record.assert_not_called()


def test_missing_local_raw_file_is_not_silent(tmp_path):
    uploader = object.__new__(S3DataUploader)
    uploader.upload_file = Mock()
    with pytest.raises(FileNotFoundError, match='Required dataset'):
        uploader.upload_bank_data(str(tmp_path))
    uploader.upload_file.assert_not_called()
