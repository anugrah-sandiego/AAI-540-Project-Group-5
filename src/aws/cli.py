"""
aws-tools: set up and operate the AWS side of the project from the laptop.

Uses the temporary AWS Academy credentials in .env; everything scheduled runs in
AWS as LabRole instead. Run inside the aws-tools container:

    docker compose run --rm aws-tools <command> [options]

    check            show identity, role and whether the scheduler may use the role
    replay           upload the next daily batches of bank-full.csv to the raw zone
    deploy-pipeline  create or update the pipeline for an image tag (optionally start it)
    approve          approve the newest pending model version (it becomes champion)
    rollback         reject the champion so the previous approved version takes over
    pull-model       download the champion for the local API
    smoke-test       deploy the champion to a serverless endpoint, score one record, delete it
    schedule         run the pipeline once at a time, daily, or turn the schedule off
"""

import argparse
import io
import json
import logging
import tarfile

import pandas as pd
from botocore.exceptions import ClientError
from sagemaker.workflow.pipeline_context import PipelineSession

from src.aws.config import MODEL_GROUP, PIPELINE_NAME, AwsConfig, session
from src.aws.pipeline import build_pipeline
from src.aws.registry import latest_packages, model_data_url
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DATA_FILE = 'data/raw/bank-full.csv'
SEPARATOR = ';'
SCHEDULE_NAME = 'bank-marketing-daily'
ENDPOINT_NAME = 'bank-marketing-smoke-test'


def batch_rows(index: int, initial: int, size: int) -> slice:
    """Rows of the date-ordered file in a batch: a large history first, then fixed-size days."""
    if index == 0:
        return slice(0, initial)
    start = initial + (index - 1) * size
    return slice(start, start + size)


def check(cfg: AwsConfig, aws) -> None:
    print(f"Account  {cfg.account}\nRegion   {cfg.region}\nRole     {cfg.role_arn}\nBucket   {cfg.bucket}")
    try:
        role = aws.client('iam').get_role(RoleName=cfg.role_arn.rsplit('/', 1)[-1])['Role']
    except ClientError as exc:
        print(f"Scheduler may use the role: unknown ({exc.response['Error']['Code']})")
        return
    trusted = 'scheduler.amazonaws.com' in json.dumps(role['AssumeRolePolicyDocument'])
    print(f"Scheduler may use the role: {'yes' if trusted else 'NO, use `deploy-pipeline --start` instead'}")


def ensure_bucket(cfg: AwsConfig, s3) -> None:
    try:
        s3.head_bucket(Bucket=cfg.bucket)
    except ClientError:
        create = {} if cfg.region == 'us-east-1' else {
            'CreateBucketConfiguration': {'LocationConstraint': cfg.region}}
        s3.create_bucket(Bucket=cfg.bucket, **create)


def replay(cfg: AwsConfig, aws, batches: int, initial: int, size: int) -> None:
    """Upload the next batches; the checkpoint file is created so the first pipeline run can read it."""
    s3 = aws.client('s3')
    ensure_bucket(cfg, s3)
    listing = s3.list_objects_v2(Bucket=cfg.bucket, Prefix=cfg.key('raw') + '/', Delimiter='/')
    next_index = len(listing.get('CommonPrefixes', []))
    data = pd.read_csv(DATA_FILE, sep=SEPARATOR)
    for index in range(next_index, next_index + batches):
        rows = data.iloc[batch_rows(index, initial, size)]
        if rows.empty:
            print("No rows left to replay")
            break
        key = cfg.key('raw', f'batch_id=b{index:04d}', 'part.csv')
        s3.put_object(Bucket=cfg.bucket, Key=key, Body=rows.to_csv(sep=SEPARATOR, index=False).encode())
        print(f"Uploaded {len(rows)} rows to s3://{cfg.bucket}/{key}")
    checkpoint_key = cfg.key('checkpoints', 'processed_batches.json')
    try:
        s3.head_object(Bucket=cfg.bucket, Key=checkpoint_key)
    except ClientError:
        s3.put_object(Bucket=cfg.bucket, Key=checkpoint_key, Body=b'{}')


def deploy_pipeline(cfg: AwsConfig, aws, image_tag: str, start: bool, full: bool) -> None:
    sagemaker = aws.client('sagemaker')
    ensure_bucket(cfg, aws.client('s3'))
    try:
        sagemaker.create_model_package_group(ModelPackageGroupName=MODEL_GROUP,
                                             ModelPackageGroupDescription='Term deposit models')
    except ClientError as exc:
        if 'already exists' not in str(exc):
            raise
    pipeline = build_pipeline(cfg, image_tag, PipelineSession(boto_session=aws))
    pipeline.upsert(role_arn=cfg.role_arn)
    print(f"Pipeline {PIPELINE_NAME} now runs image {cfg.image_uri(image_tag)}")
    if start:
        parameters = {'QuickTraining': 'false'} if full else {}
        print(f"Started {pipeline.start(parameters=parameters).arn}")


def approve(aws) -> None:
    sagemaker = aws.client('sagemaker')
    pending = latest_packages(sagemaker, MODEL_GROUP, 'PendingManualApproval')
    if not pending:
        raise SystemExit("No model version is waiting for approval")
    sagemaker.update_model_package(ModelPackageArn=pending[0], ModelApprovalStatus='Approved')
    print(f"Approved {pending[0]}; it is now the champion")


def rollback(aws) -> None:
    sagemaker = aws.client('sagemaker')
    approved = latest_packages(sagemaker, MODEL_GROUP, 'Approved', limit=2)
    if len(approved) < 2:
        raise SystemExit("No earlier approved version to roll back to")
    sagemaker.update_model_package(ModelPackageArn=approved[0], ModelApprovalStatus='Rejected',
                                   ApprovalDescription='Rolled back')
    print(f"Rejected {approved[0]}; champion is now {approved[1]}")


def champion(sagemaker) -> str:
    approved = latest_packages(sagemaker, MODEL_GROUP, 'Approved')
    if not approved:
        raise SystemExit("No approved model yet; run `approve` first")
    return approved[0]


def pull_model(aws, dest: str) -> None:
    sagemaker = aws.client('sagemaker')
    package = champion(sagemaker)
    bucket, key = model_data_url(sagemaker, package).removeprefix('s3://').split('/', 1)
    body = aws.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read()
    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        archive.extractall(dest, filter='data')
    print(f"Champion {package} extracted to {dest}/")


def smoke_test(cfg: AwsConfig, aws) -> None:
    """Serverless keeps the cost to the seconds the test runs; everything is deleted afterwards."""
    sagemaker = aws.client('sagemaker')
    package = champion(sagemaker)
    record = pd.read_csv(DATA_FILE, sep=SEPARATOR, nrows=1).drop(columns=['y', 'duration'])
    sagemaker.create_model(ModelName=ENDPOINT_NAME, ExecutionRoleArn=cfg.role_arn,
                           Containers=[{'ModelPackageName': package}])
    try:
        sagemaker.create_endpoint_config(EndpointConfigName=ENDPOINT_NAME, ProductionVariants=[{
            'VariantName': 'AllTraffic', 'ModelName': ENDPOINT_NAME,
            'ServerlessConfig': {'MemorySizeInMB': 2048, 'MaxConcurrency': 1}}])
        sagemaker.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_NAME)
        print("Waiting for the endpoint (a few minutes)...")
        sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=ENDPOINT_NAME)
        response = aws.client('sagemaker-runtime').invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='application/json',
            Body=json.dumps({'instances': record.to_dict(orient='records')}))
        print(json.dumps(json.loads(response['Body'].read()), indent=2))
    finally:
        for delete, name in ((sagemaker.delete_endpoint, 'EndpointName'),
                             (sagemaker.delete_endpoint_config, 'EndpointConfigName'),
                             (sagemaker.delete_model, 'ModelName')):
            try:
                delete(**{name: ENDPOINT_NAME})
            except ClientError:
                pass
        print("Endpoint, endpoint config and model deleted")


def schedule(cfg: AwsConfig, aws, once: str, daily: bool, off: bool) -> None:
    scheduler = aws.client('scheduler')
    if off:
        scheduler.delete_schedule(Name=SCHEDULE_NAME)
        print(f"Deleted schedule {SCHEDULE_NAME}")
        return
    pipeline_arn = aws.client('sagemaker').describe_pipeline(PipelineName=PIPELINE_NAME)['PipelineArn']
    target = {'Arn': pipeline_arn, 'RoleArn': cfg.role_arn,
              'SageMakerPipelineParameters': {'PipelineParameterList': []},
              'RetryPolicy': {'MaximumRetryAttempts': 2, 'MaximumEventAgeInSeconds': 3600}}
    expression = 'cron(0 6 * * ? *)' if daily else f'at({once})'
    extra = {} if daily else {'ActionAfterCompletion': 'DELETE'}
    scheduler.create_schedule(Name=SCHEDULE_NAME, ScheduleExpression=expression,
                              ScheduleExpressionTimezone='UTC', FlexibleTimeWindow={'Mode': 'OFF'},
                              Target=target, **extra)
    print(f"Schedule {SCHEDULE_NAME}: {expression} UTC")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog='aws-tools', description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    replay_args = commands.add_parser('replay')
    replay_args.add_argument('--batches', type=int, default=1)
    # The file is in date order and the subscription rate climbs from 3% to 47%; starting
    # with 70% of it as history gives the first model enough subscribers to learn from
    replay_args.add_argument('--initial-rows', type=int, default=31500)
    replay_args.add_argument('--batch-rows', type=int, default=350)
    deploy_args = commands.add_parser('deploy-pipeline')
    deploy_args.add_argument('--image-tag', required=True, help='tag printed by scripts/push_image.sh')
    deploy_args.add_argument('--start', action='store_true')
    deploy_args.add_argument('--full', action='store_true', help='with --start: compare all 12 models')
    commands.add_parser('approve')
    commands.add_parser('rollback')
    pull_args = commands.add_parser('pull-model')
    pull_args.add_argument('--dest', default='models')
    commands.add_parser('smoke-test')
    schedule_args = commands.add_parser('schedule').add_mutually_exclusive_group(required=True)
    schedule_args.add_argument('--once', metavar='YYYY-MM-DDTHH:MM:SS', help='one run at this UTC time')
    schedule_args.add_argument('--daily', action='store_true', help='every day at 06:00 UTC')
    schedule_args.add_argument('--off', action='store_true')
    args = parser.parse_args()

    aws = session()
    cfg = AwsConfig.from_env(aws)
    if args.command == 'check':
        check(cfg, aws)
    elif args.command == 'replay':
        replay(cfg, aws, args.batches, args.initial_rows, args.batch_rows)
    elif args.command == 'deploy-pipeline':
        deploy_pipeline(cfg, aws, args.image_tag, args.start, args.full)
    elif args.command == 'approve':
        approve(aws)
    elif args.command == 'rollback':
        rollback(aws)
    elif args.command == 'pull-model':
        pull_model(aws, args.dest)
    elif args.command == 'smoke-test':
        smoke_test(cfg, aws)
    else:
        schedule(cfg, aws, args.once, args.daily, args.off)


if __name__ == '__main__':
    main()
