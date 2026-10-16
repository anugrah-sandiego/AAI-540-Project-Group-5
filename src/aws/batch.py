"""
Batch scoring with the champion: a SageMaker Batch Transform job on the project image.

Customers are written as JSON Lines, one `{"instances": [...]}` request per line,
so the job sends each line to the same /invocations handler the API uses and
gets one JSON response per line back. Nothing keeps running after the job.
Adapted from Anugrah Rastogi's src/aws/sagemaker/batch_inference.py on main.
"""

import json
import time
from typing import List

import pandas as pd

from src.aws.config import MODEL_GROUP, AwsConfig
from src.aws.registry import latest_packages

CHUNK = 500  # customers per request line


def to_requests(customers: pd.DataFrame) -> str:
    """One JSON request line per chunk of customers."""
    lines = [json.dumps({'instances': customers.iloc[start:start + CHUNK].to_dict(orient='records')})
             for start in range(0, len(customers), CHUNK)]
    return '\n'.join(lines) + '\n'


def from_responses(text: str) -> List[float]:
    """Probabilities from the response lines, in request order."""
    return [p['probability'] for line in text.splitlines() if line.strip()
            for p in json.loads(line)['predictions']]


def batch_score(cfg: AwsConfig, aws, customers: pd.DataFrame, instance_type: str = 'ml.m5.large') -> pd.DataFrame:
    """Score customers with the champion and return them with a `probability` column."""
    sagemaker, s3 = aws.client('sagemaker'), aws.client('s3')
    packages = latest_packages(sagemaker, MODEL_GROUP, 'Approved')
    if not packages:
        raise SystemExit("No approved model yet; run `approve` first")
    job = f'bank-marketing-batch-{time.strftime("%Y%m%d-%H%M%S")}'
    input_key = cfg.key('batch', job, 'input', 'customers.jsonl')
    s3.put_object(Bucket=cfg.bucket, Key=input_key, Body=to_requests(customers).encode())

    sagemaker.create_model(ModelName=job, ExecutionRoleArn=cfg.role_arn,
                           Containers=[{'ModelPackageName': packages[0]}])
    try:
        sagemaker.create_transform_job(
            TransformJobName=job, ModelName=job, BatchStrategy='SingleRecord',
            TransformInput={'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
                                                            'S3Uri': cfg.s3('batch', job, 'input')}},
                            'ContentType': 'application/json', 'SplitType': 'Line'},
            TransformOutput={'S3OutputPath': cfg.s3('batch', job, 'output'), 'Accept': 'application/json',
                             'AssembleWith': 'Line'},
            TransformResources={'InstanceType': instance_type, 'InstanceCount': 1})
        print(f"Batch transform {job} started; waiting (a few minutes)...")
        sagemaker.get_waiter('transform_job_completed_or_stopped').wait(TransformJobName=job)
        status = sagemaker.describe_transform_job(TransformJobName=job)
        if status['TransformJobStatus'] != 'Completed':
            raise SystemExit(f"Batch transform {job} {status['TransformJobStatus']}: {status.get('FailureReason')}")
        output = s3.get_object(Bucket=cfg.bucket, Key=cfg.key('batch', job, 'output', 'customers.jsonl.out'))
        return customers.assign(probability=from_responses(output['Body'].read().decode()))
    finally:
        sagemaker.delete_model(ModelName=job)
