"""
Real-time endpoint for the champion, with data capture, Model Monitor and a dashboard.

`endpoint --up` deploys the latest approved model on one instance, captures every
request, sends sample traffic, sets a data-quality baseline from the training
split and an hourly monitoring schedule, and creates a CloudWatch dashboard.
An instance endpoint bills every hour it exists, so `endpoint --down` removes the
schedule, endpoint, endpoint config, model and dashboard again.
Adapted from Anugrah Rastogi's deployment.py and monitoring.py on main.
"""

import json
import time
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.session import Session

from src.aws.config import DASHBOARD_NAME, ENDPOINT_NAME, MODEL_GROUP, MONITOR_SCHEDULE, AwsConfig
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
    sagemaker, s3 = aws.client('sagemaker'), aws.client('s3')
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

    # Sample traffic from the training split, so capture, monitoring and the dashboard have data
    train = model_inputs(pd.read_parquet(TRAIN_SPLIT))
    runtime = aws.client('sagemaker-runtime')
    for record in train.sample(SAMPLE_REQUESTS, random_state=0).to_dict(orient='records'):
        runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json',
                                Body=json.dumps({'instances': [record]}))
    print(f"Sent {SAMPLE_REQUESTS} sample requests")

    aws.client('cloudwatch').put_dashboard(DashboardName=DASHBOARD_NAME, DashboardBody=dashboard_body(cfg.region))
    print(f"CloudWatch dashboard {DASHBOARD_NAME} created")

    s3.put_object(Bucket=cfg.bucket, Key=cfg.key('monitoring', 'baseline', 'train.csv'),
                  Body=train.to_csv(index=False).encode())
    s3.put_object(Bucket=cfg.bucket, Key=cfg.key('monitoring', 'preprocessor.py'),
                  Body=(Path(__file__).parent / 'monitor_preprocessor.py').read_bytes())
    monitor = DefaultModelMonitor(role=cfg.role_arn, instance_count=1, instance_type='ml.m5.large',
                                  volume_size_in_gb=20, max_runtime_in_seconds=1800,
                                  sagemaker_session=Session(boto_session=aws))
    print("Computing the data-quality baseline from the training split (about 5 minutes)...")
    monitor.suggest_baseline(baseline_dataset=cfg.s3('monitoring', 'baseline', 'train.csv'),
                             dataset_format=DatasetFormat.csv(header=True),
                             output_s3_uri=cfg.s3('monitoring', 'baseline', 'results'), wait=True, logs=False)
    monitor.create_monitoring_schedule(
        monitor_schedule_name=MONITOR_SCHEDULE, endpoint_input=ENDPOINT_NAME,
        record_preprocessor_script=cfg.s3('monitoring', 'preprocessor.py'),
        output_s3_uri=cfg.s3('monitoring', 'reports'), statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(), schedule_cron_expression=CronExpressionGenerator.hourly())
    print(f"Monitoring schedule {MONITOR_SCHEDULE} runs hourly; reports go to {cfg.s3('monitoring', 'reports')}")


def endpoint_down(aws) -> None:
    """Remove everything `endpoint --up` created; safe to run when parts are already gone."""
    sagemaker = aws.client('sagemaker')
    steps = [(sagemaker.delete_monitoring_schedule, {'MonitoringScheduleName': MONITOR_SCHEDULE}),
             (sagemaker.delete_endpoint, {'EndpointName': ENDPOINT_NAME}),
             (sagemaker.delete_endpoint_config, {'EndpointConfigName': ENDPOINT_NAME}),
             (sagemaker.delete_model, {'ModelName': ENDPOINT_NAME}),
             (aws.client('cloudwatch').delete_dashboards, {'DashboardNames': [DASHBOARD_NAME]})]
    for delete, arguments in steps:
        try:
            delete(**arguments)
            print(f"Deleted {next(iter(arguments.values()))}")
        except ClientError as exc:
            print(f"Skipped {next(iter(arguments.values()))}: {exc.response['Error']['Code']}")
        if delete == sagemaker.delete_monitoring_schedule:
            wait_until_schedule_gone(sagemaker)


def wait_until_schedule_gone(sagemaker, attempts: int = 30) -> None:
    """The endpoint cannot be deleted while its monitoring schedule still exists."""
    for _ in range(attempts):
        try:
            sagemaker.describe_monitoring_schedule(MonitoringScheduleName=MONITOR_SCHEDULE)
        except ClientError:
            return
        time.sleep(10)
