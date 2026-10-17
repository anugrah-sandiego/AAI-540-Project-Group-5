"""CI adapters for the project's original CLI modules.

No clients are constructed on import. GROUP5_CI=1 selects this path from the
original modules without changing their legacy importable class APIs.
"""
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd
from cicd import release
from src.data_ingestion.data_ingestion import validate_schema


def feature_definitions(frame):
    definitions = []
    for name in frame.columns:
        dtype = frame[name].dtype
        if pd.api.types.is_integer_dtype(dtype):
            kind = 'Integral'
        elif pd.api.types.is_float_dtype(dtype):
            kind = 'Fractional'
        else:
            kind = 'String'
        definitions.append({'FeatureName': name, 'FeatureType': kind})
    return definitions


def validate_group(group, definitions):
    if group['FeatureGroupStatus'] != 'Created':
        raise RuntimeError('Feature group is not ready: ' + group['FeatureGroupStatus'])
    if group['RecordIdentifierFeatureName'] != 'customer_id' or group['EventTimeFeatureName'] != 'event_time':
        raise ValueError('Existing feature group uses different identifier/event-time columns')
    actual = {f['FeatureName']: f['FeatureType'] for f in group['FeatureDefinitions']}
    expected = {f['FeatureName']: f['FeatureType'] for f in definitions}
    if actual != expected:
        raise ValueError('Existing feature group schema differs. Use a new group name or explicitly migrate it.')
    if not group.get('OfflineStoreConfig', {}).get('S3StorageConfig'):
        raise ValueError('Feature group must have an offline store')


def wait_group(sm, name):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        group = sm.describe_feature_group(FeatureGroupName=name)
        if group['FeatureGroupStatus'] == 'Created':
            return group
        if group['FeatureGroupStatus'] != 'Creating':
            raise RuntimeError('Feature group creation failed: ' + str(group.get('FailureReason', group['FeatureGroupStatus'])))
        time.sleep(10)
    raise TimeoutError('Feature group not ready after 15 minutes')


def sync_feature_store():
    raw = release.read_s3(os.environ['RAW_DATA_URI'])
    digest = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(io.BytesIO(raw), sep=';')
    validate_schema(frame)
    if frame.empty or frame.duplicated().any():
        raise ValueError('Feature Store input must be nonempty without duplicate rows')
    # This is a dataset-row identifier, not a real bank customer identity.
    frame['customer_id'] = [f'{digest[:16]}-{i}' for i in range(len(frame))]
    frame['event_time'] = float(time.time())
    definitions = feature_definitions(frame)
    region = os.environ['AWS_REGION']
    sm = release.client('sagemaker')
    s3 = release.client('s3')
    bucket = os.environ['AWS_BUCKET']
    name = os.environ.get('FEATURE_GROUP_NAME', 'bank-marketing-features')
    existing = False
    try:
        group = sm.describe_feature_group(FeatureGroupName=name)
        existing = True
    except ClientError as exc:
        if not release.missing(exc):
            raise
        sm.create_feature_group(FeatureGroupName=name,
            RecordIdentifierFeatureName='customer_id', EventTimeFeatureName='event_time',
            FeatureDefinitions=definitions, RoleArn=os.environ['AWS_ROLE_ARN'],
            OfflineStoreConfig={'S3StorageConfig': {
                'S3Uri': f's3://{bucket}/aai540/bank-marketing/feature-store/'}})
        group = wait_group(sm, name)
    if group['FeatureGroupStatus'] == 'Creating':
        group = wait_group(sm, name)
    validate_group(group, definitions)
    arn = group['FeatureGroupArn']
    # Group creation time distinguishes a recreated group from an old deleted group.
    group_generation = hashlib.sha256(str(group['CreationTime']).encode()).hexdigest()[:12]
    marker = f'aai540/bank-marketing/cicd/feature-store-checkpoints/{name}/{group_generation}/{digest}.json'
    if existing and os.environ.get('FORCE_FEATURE_INGESTION') != '1':
        try:
            checkpoint = json.loads(s3.get_object(Bucket=bucket, Key=marker)['Body'].read())
        except ClientError as exc:
            if exc.response['Error']['Code'] not in ('NoSuchKey', '404', 'NotFound'):
                raise
        else:
            if checkpoint.get('source_sha256') != digest or checkpoint.get('rows') != len(frame):
                raise ValueError('Invalid Feature Store checkpoint')
            write_feature_state(dict(checkpoint, status='unchanged_dataset_skipped'))
            print('Feature Store: unchanged dataset already ingested; skipping', name)
            return
    runtime = boto3.client('sagemaker-featurestore-runtime', region_name=region,
                           config=Config(retries={'mode': 'standard', 'max_attempts': 10}))
    kinds = {f['FeatureName']: f['FeatureType'] for f in definitions}
    columns = list(frame.columns)
    def put(values):
        record = []
        for column, value in zip(columns, values):
            encoded = str(int(value)) if kinds[column] == 'Integral' else str(value)
            record.append({'FeatureName': column, 'ValueAsString': encoded})
        runtime.put_record(FeatureGroupName=name, Record=record, TargetStores=['OfflineStore'])
    workers = int(os.environ.get('FEATURE_INGEST_WORKERS', '4'))
    if not 1 <= workers <= 16:
        raise ValueError('FEATURE_INGEST_WORKERS must be between 1 and 16')
    # Consume all futures so a failed ingestion never creates a completion checkpoint.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(put, frame.itertuples(index=False, name=None)):
            pass
    state = {'feature_group': name, 'feature_group_arn': arn, 'source_sha256': digest,
             'rows': len(frame), 'status': 'ingested',
             'offline_store': group['OfflineStoreConfig']['S3StorageConfig']}
    s3.put_object(Bucket=bucket, Key=marker, Body=json.dumps(state).encode())
    write_feature_state(state)
    print('Feature Store ingestion accepted:', len(frame), 'records;', arn)


def write_feature_state(state):
    folder = Path('artifacts'); folder.mkdir(exist_ok=True)
    (folder / 'feature_store.json').write_text(json.dumps(state, indent=2, default=str))


def export_training_outputs(state):
    outputs = {'TRAINING_JOB_NAME': state['training_job'], 'MODEL_ARTIFACT_URI': state['model_uri']}
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            for name, value in outputs.items():
                stream.write(f'{name}={value}\n')
    for name, value in outputs.items():
        print(name + '=' + value)


def run(stage):
    if stage == 'feature_store':
        sync_feature_store()
    elif stage == 'training':
        release.prepare()
        release.train()
        export_training_outputs(release.load())
    elif stage == 'model_registry':
        release.register()  # refuses candidates without a successful evaluation gate
    elif stage == 'deployment':
        # This entrypoint is called only by the protected production job in the workflow.
        release.approve()
        release.staging()
        release.promote()
    else:
        raise ValueError('Unknown CI module stage: ' + stage)
