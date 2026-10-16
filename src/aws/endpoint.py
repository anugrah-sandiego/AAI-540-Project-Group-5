"""
Real-time endpoint for the champion, with data capture and a CloudWatch dashboard.

`endpoint --up` deploys the latest approved model on one instance, captures every
request to S3, sends sample traffic and creates a CloudWatch dashboard;
`aws-tools drift-report` then compares the captured requests with the training
data (SageMaker Model Monitor's data-quality jobs are closed to new customers).
An instance endpoint bills every hour it exists, so `endpoint --down` removes the
endpoint, endpoint config, model and dashboard again.
Adapted from Anugrah Rastogi's deployment.py and monitoring.py on main.
"""

import json
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError

from src.aws.config import DASHBOARD_NAME, ENDPOINT_NAME, MODEL_GROUP, AwsConfig
from src.aws.registry import latest_packages

TRAIN_SPLIT = Path('data/processed/train.parquet')
SAMPLE_REQUESTS = 50
METRICS = [('Invocations', 'Sum'), ('ModelLatency', 'Average'), ('Invocation4XXErrors', 'Sum'),
           ('Invocation5XXErrors', 'Sum')]


def dashboard_body(region: str) -> str:
    """CloudWatch widgets for the endpoint's traffic, latency and errors."""
    widgets = [{'type': 'metric', 'width': 12, 'height': 6, 'x': 12 * (i % 2), 'y': 6 * (i // 2),
                'properties': {'title': f'{metric} ({stat})', 'region': region, 'stat': stat, 'period': 300,
                               'metrics': [['AWS/SageMaker', metric, 'EndpointName', ENDPOINT_NAME,
                                            'VariantName', 'AllTraffic']]}}
               for i, (metric, stat) in enumerate(METRICS)]
    return json.dumps({'widgets': widgets})


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=['y', 'duration'], errors='ignore')


def endpoint_up(cfg: AwsConfig, aws, instance_type: str) -> None:
    sagemaker = aws.client('sagemaker')
    packages = latest_packages(sagemaker, MODEL_GROUP, 'Approved')
    if not packages:
        raise SystemExit("No approved model yet; run `approve` first")
    sagemaker.create_model(ModelName=ENDPOINT_NAME, ExecutionRoleArn=cfg.role_arn,
                           Containers=[{'ModelPackageName': packages[0]}])
    sagemaker.create_endpoint_config(
        EndpointConfigName=ENDPOINT_NAME,
        ProductionVariants=[{'VariantName': 'AllTraffic', 'ModelName': ENDPOINT_NAME,
                             'InstanceType': instance_type, 'InitialInstanceCount': 1}],
        DataCaptureConfig={'EnableCapture': True, 'InitialSamplingPercentage': 100,
                           'DestinationS3Uri': cfg.s3('monitoring', 'capture'),
                           'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],
                           'CaptureContentTypeHeader': {'JsonContentTypes': ['application/json']}})
    sagemaker.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_NAME)
    print(f"Creating endpoint {ENDPOINT_NAME} on {instance_type} for {packages[0]} (5-10 minutes)...")
    sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=ENDPOINT_NAME)
    print("Endpoint in service. It bills every hour until `aws-tools endpoint --down`.")

    # Sample traffic from the training split, so capture, the drift report and the dashboard have data
    train = model_inputs(pd.read_parquet(TRAIN_SPLIT))
    runtime = aws.client('sagemaker-runtime')
    for record in train.sample(SAMPLE_REQUESTS, random_state=0).to_dict(orient='records'):
        runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json',
                                Body=json.dumps({'instances': [record]}))
    print(f"Sent {SAMPLE_REQUESTS} sample requests")

    aws.client('cloudwatch').put_dashboard(DashboardName=DASHBOARD_NAME, DashboardBody=dashboard_body(cfg.region))
    print(f"CloudWatch dashboard {DASHBOARD_NAME} created")
    print("Check drift with `aws-tools drift-report`; remove everything with `aws-tools endpoint --down`.")


def endpoint_down(aws) -> None:
    """Remove everything `endpoint --up` created; safe to run when parts are already gone."""
    sagemaker = aws.client('sagemaker')
    steps = [(sagemaker.delete_endpoint, {'EndpointName': ENDPOINT_NAME}),
             (sagemaker.delete_endpoint_config, {'EndpointConfigName': ENDPOINT_NAME}),
             (sagemaker.delete_model, {'ModelName': ENDPOINT_NAME}),
             (aws.client('cloudwatch').delete_dashboards, {'DashboardNames': [DASHBOARD_NAME]})]
    for delete, arguments in steps:
        try:
            delete(**arguments)
            print(f"Deleted {next(iter(arguments.values()))}")
        except ClientError as exc:
            print(f"Skipped {next(iter(arguments.values()))}: {exc.response['Error']['Code']}")
