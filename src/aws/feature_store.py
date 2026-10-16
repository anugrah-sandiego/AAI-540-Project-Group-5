"""
SageMaker Feature Store for the incoming customer batches.

PrepareData writes every new batch to an offline feature group, so each
accepted batch is kept with its arrival time and can be queried later (Athena).
Only boto3 is used, so the pipeline image can write without the SageMaker SDK.
Group creation (schema check, wait until ready) is adapted from Anugrah Rastogi's
src/aws/sagemaker/feature_store.py on main.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pandas as pd
from botocore.exceptions import ClientError

from src.aws.config import FEATURE_GROUP, AwsConfig

RECORD_ID, EVENT_TIME = 'record_id', 'event_time'
# Pre-call inputs and the outcome; `duration` is only known after the call, so it is left out
FEATURES = {'age': 'Integral', 'job': 'String', 'marital': 'String', 'education': 'String',
            'default': 'String', 'balance': 'Integral', 'housing': 'String', 'loan': 'String',
            'contact': 'String', 'day': 'Integral', 'month': 'String', 'campaign': 'Integral',
            'pdays': 'Integral', 'previous': 'Integral', 'poutcome': 'String', 'y': 'String'}
DEFINITIONS = ([{'FeatureName': RECORD_ID, 'FeatureType': 'String'},
                {'FeatureName': EVENT_TIME, 'FeatureType': 'String'},
                {'FeatureName': 'batch_id', 'FeatureType': 'String'}]
               + [{'FeatureName': name, 'FeatureType': kind} for name, kind in FEATURES.items()])


def ensure_feature_group(cfg: AwsConfig, sagemaker, timeout: int = 900) -> None:
    """Create the offline feature group, or check that an existing one has the expected schema."""
    try:
        existing = sagemaker.describe_feature_group(FeatureGroupName=FEATURE_GROUP)
    except ClientError as exc:
        if exc.response['Error']['Code'] not in ('ResourceNotFound', 'ResourceNotFoundException'):
            raise
        sagemaker.create_feature_group(
            FeatureGroupName=FEATURE_GROUP, RecordIdentifierFeatureName=RECORD_ID,
            EventTimeFeatureName=EVENT_TIME, FeatureDefinitions=DEFINITIONS, RoleArn=cfg.role_arn,
            OfflineStoreConfig={'S3StorageConfig': {'S3Uri': cfg.s3('feature-store')}},
            Description='Customer batches accepted by the pipeline (pre-call inputs and outcome)')
        existing = {'FeatureGroupStatus': 'Creating'}
    else:
        actual = {d['FeatureName']: d['FeatureType'] for d in existing['FeatureDefinitions']}
        if actual != {d['FeatureName']: d['FeatureType'] for d in DEFINITIONS}:
            raise ValueError(f'Feature group {FEATURE_GROUP} exists with a different schema')
    deadline = time.monotonic() + timeout
    while existing['FeatureGroupStatus'] == 'Creating':
        if time.monotonic() > deadline:
            raise TimeoutError(f'Feature group {FEATURE_GROUP} is still being created')
        time.sleep(10)
        existing = sagemaker.describe_feature_group(FeatureGroupName=FEATURE_GROUP)
    if existing['FeatureGroupStatus'] != 'Created':
        raise RuntimeError(f"Feature group {FEATURE_GROUP} is {existing['FeatureGroupStatus']}")


def records(batch: pd.DataFrame, batch_id: str, event_time: str) -> List[List[Dict[str, str]]]:
    """One Feature Store record per customer: an id unique to the batch row, plus every feature."""
    rows = []
    for position, customer in enumerate(batch[list(FEATURES)].to_dict(orient='records')):
        values = {RECORD_ID: f'{batch_id}-{position:05d}', EVENT_TIME: event_time, 'batch_id': batch_id, **customer}
        rows.append([{'FeatureName': name, 'ValueAsString': str(value)} for name, value in values.items()])
    return rows


def put_batch(runtime, batch: pd.DataFrame, batch_id: str, event_time: str, workers: int = 8) -> int:
    """Write a batch to the offline store; returns the number of records written."""
    def put(record):
        runtime.put_record(FeatureGroupName=FEATURE_GROUP, Record=record, TargetStores=['OfflineStore'])
    batch_records = records(batch, batch_id, event_time)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, batch_records))
    return len(batch_records)
