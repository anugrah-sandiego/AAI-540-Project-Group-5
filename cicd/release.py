"""Explicit release stages. No AWS client is constructed on import."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
from urllib.parse import urlparse
import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from cicd.container.train import separate

STATE = Path('artifacts/release.json')


def save(state):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def load():
    return json.loads(STATE.read_text())


def client(service):
    return boto3.client(service, region_name=os.environ.get('AWS_REGION', 'us-east-1'))


def location(uri):
    parsed = urlparse(uri)
    if parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/'):
        raise ValueError('An S3 object URI is required')
    return parsed.netloc, parsed.path.lstrip('/')


def read_s3(uri):
    bucket, key = location(uri)
    return client('s3').get_object(Bucket=bucket, Key=key)['Body'].read()


def missing(exc):
    error = exc.response['Error']
    return error['Code'] in ('ResourceNotFound', 'ResourceNotFoundException') or (
        error['Code'] == 'ValidationException' and any(
            s in error.get('Message', '').lower() for s in ('could not find', 'not found', 'does not exist')))


def gates(values, minimum_auc=.75, minimum_lift=2.5):
    for name, minimum in [('roc_auc', minimum_auc), ('lift_at_10', minimum_lift)]:
        value = float(values[name])
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f'Quality gate failed: {name}={value}, required >= {minimum}')


def prepare():
    """Read raw S3 data, validate, then write immutable, disjoint split objects."""
    from src.data_ingestion.data_ingestion import validate_schema
    raw_uri = os.environ['RAW_DATA_URI']
    raw = read_s3(raw_uri)
    df = pd.read_csv(io.BytesIO(raw), sep=';')
    validate_schema(df)
    if df.duplicated().any():
        raise ValueError('Duplicate raw rows: resolve before splitting')
    separate(df)  # fail before any training when labels are invalid
    train, holdout = train_test_split(df, test_size=.30, stratify=df.y, random_state=42)
    val, test = train_test_split(holdout, test_size=.50, stratify=holdout.y, random_state=42)
    bucket = os.environ['AWS_BUCKET']
    run = os.environ['RELEASE_ID']
    if not run.replace('-', '').isalnum() or len(run) > 40:
        raise ValueError('RELEASE_ID must be <=40 alphanumeric/hyphen characters')
    root = f'aai540/bank-marketing/cicd/{run}'
    state = {'run': run, 'bucket': bucket, 'root': root,
             'source_uri': raw_uri, 'source_sha256': hashlib.sha256(raw).hexdigest(),
             'commit': os.environ.get('GITHUB_SHA', 'local'), 'inputs': {}}
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        buf = io.BytesIO(); frame.to_parquet(buf, index=False)
        key = f'{root}/processed/{name}.parquet'
        client('s3').put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        state['inputs'][name] = f's3://{bucket}/{key}'
    state['split_rows'] = [len(train), len(val), len(test)]
    save(state)


def train():
    s = load(); sm = client('sagemaker')
    # Resolve the framework image from a completed job in this account, or an explicit URI.
    image = os.environ.get('TRAINING_IMAGE_URI')
    if not image:
        base = sm.describe_training_job(TrainingJobName=os.environ['BASE_TRAINING_JOB_NAME'])
        if base['TrainingJobStatus'] != 'Completed':
            raise ValueError('Image reference job must be Completed')
        image = base['AlgorithmSpecification']['TrainingImage']
    code = io.BytesIO()
    with tarfile.open(fileobj=code, mode='w:gz') as archive:
        for path in sorted(Path('cicd/container').glob('*')):
            if path.is_file() and path.suffix in ('.py', '.txt'):
                archive.add(path, arcname=path.name)
    key = s['root'] + '/source/sourcedir.tar.gz'
    client('s3').put_object(Bucket=s['bucket'], Key=key, Body=code.getvalue())
    source = f"s3://{s['bucket']}/{key}"
    s['training_job'] = 'bank-ci-' + s['run']; save(s)
    sm.create_training_job(TrainingJobName=s['training_job'], RoleArn=os.environ['AWS_ROLE_ARN'],
        AlgorithmSpecification={'TrainingImage': image, 'TrainingInputMode': 'File'},
        HyperParameters={'sagemaker_program': json.dumps('train.py'),
            'sagemaker_submit_directory': json.dumps(source),
            'sagemaker_container_log_level': '20',
            'sagemaker_region': json.dumps(os.environ['AWS_REGION'])},
        InputDataConfig=[{'ChannelName': name,
            'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix', 'S3Uri': uri,
                                           'S3DataDistributionType': 'FullyReplicated'}},
            'ContentType': 'application/x-parquet'} for name, uri in s['inputs'].items()],
        OutputDataConfig={'S3OutputPath': f"s3://{s['bucket']}/{s['root']}/models"},
        ResourceConfig={'InstanceType': 'ml.m5.xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 30},
        StoppingCondition={'MaxRuntimeInSeconds': 3600})
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=s['training_job'],
        WaiterConfig={'Delay': 30, 'MaxAttempts': 180})
    job = sm.describe_training_job(TrainingJobName=s['training_job'])
    if job['TrainingJobStatus'] != 'Completed':
        raise RuntimeError('Training did not complete: ' + job['TrainingJobStatus'])
    s['model_uri'] = job['ModelArtifacts']['S3ModelArtifacts']
    s['image'] = image
    save(s)


def evaluate():
    s = load()
    # Only JSON members are read. Never extract an archive or deserialize an untrusted pickle.
    with tarfile.open(fileobj=io.BytesIO(read_s3(s['model_uri'])), mode='r:gz') as archive:
        def member(name):
            matches = [m for m in archive.getmembers() if m.name.lstrip('./') == name and m.isfile()]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate artifact member ' + name)
            return json.load(archive.extractfile(matches[0]))
        s['evaluation'] = member('evaluation.json')
        s['smoke'] = member('smoke.json')
    save(s)  # retain rejected evaluation evidence too
    gates(s['evaluation']['test'], float(os.environ.get('MIN_ROC_AUC', '.75')),
          float(os.environ.get('MIN_LIFT_AT_10', '2.5')))
    s['gates_passed'] = True
    save(s)


def register():
    s = load()
    if not s.get('gates_passed'):
        raise ValueError('Evaluation must pass before registration')
    sm = client('sagemaker'); group = 'bank-marketing-models'
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group)
    except ClientError as exc:
        if not missing(exc):
            raise
        sm.create_model_package_group(ModelPackageGroupName=group)
    result = sm.create_model_package(ModelPackageGroupName=group,
        ClientToken=s['run'], ModelApprovalStatus='PendingManualApproval',
        ModelPackageDescription=f"CI/CD {s['commit']} from {s['training_job']}",
        InferenceSpecification={'Containers': [{'Image': s['image'], 'ModelDataUrl': s['model_uri'],
            'Environment': {'SAGEMAKER_PROGRAM': 'inference.py',
                            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code'}}],
            'SupportedContentTypes': ['application/json'],
            'SupportedResponseMIMETypes': ['application/json']},
        CustomerMetadataProperties={'commit': s['commit'], 'source_sha256': s['source_sha256']})
    s['package_arn'] = result['ModelPackageArn']; save(s)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = sm.describe_model_package(ModelPackageName=s['package_arn'])['ModelPackageStatus']
        if status == 'Completed':
            return
        if status == 'Failed':
            raise RuntimeError('Package registration failed')
        time.sleep(10)
    raise TimeoutError('Package registration pending')


def approve():
    s = load()
    client('sagemaker').update_model_package(ModelPackageArn=s['package_arn'],
        ModelApprovalStatus='Approved', ApprovalDescription='Approved through protected GitHub production environment')


def endpoint_info(name):
    try:
        return client('sagemaker').describe_endpoint(EndpointName=name)
    except ClientError as exc:
        if missing(exc):
            return None
        raise


def wait(name, config=None):
    sm = client('sagemaker')
    sm.get_waiter('endpoint_in_service').wait(EndpointName=name,
        WaiterConfig={'Delay': 30, 'MaxAttempts': 180})
    info = sm.describe_endpoint(EndpointName=name)
    if config and info['EndpointConfigName'] != config:
        raise RuntimeError('Endpoint reverted or did not reach the candidate configuration')


def config(s, kind):
    sm = client('sagemaker')
    name = f"bank-{kind}-{s['run']}"
    s[kind + '_model'] = name; s[kind + '_config'] = name
    save(s)  # record attempted resources before mutation for cleanup/recovery
    sm.create_model(ModelName=name, ExecutionRoleArn=os.environ['AWS_ROLE_ARN'],
                    PrimaryContainer={'ModelPackageName': s['package_arn']})
    sm.create_endpoint_config(EndpointConfigName=name, ProductionVariants=[{
        'VariantName': 'AllTraffic', 'ModelName': name, 'InitialInstanceCount': 2 if kind == 'prod' else 1,
        'InstanceType': 'ml.m5.xlarge', 'InitialVariantWeight': 1.0}],
        DataCaptureConfig={'EnableCapture': True, 'InitialSamplingPercentage': 100,
            'DestinationS3Uri': f"s3://{s['bucket']}/{s['root']}/capture",
            'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],
            'CaptureContentTypeHeader': {'JsonContentTypes': ['application/json']}})
    return name


def smoke(name, fixture):
    response = client('sagemaker-runtime').invoke_endpoint(EndpointName=name,
        ContentType='application/json', Accept='application/json', Body=json.dumps(fixture['request']))
    result = json.loads(response['Body'].read())
    actual = np.asarray(result['probabilities'], dtype=float)
    expected = np.asarray(fixture['expected']['probabilities'])
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise ValueError('Endpoint/offline prediction mismatch')
    if result['predictions'] != fixture['expected']['predictions']:
        raise ValueError('Endpoint labels differ from offline labels')


def staging():
    s = load(); name = config(s, 'stage')
    s['stage_endpoint'] = name; save(s)
    client('sagemaker').create_endpoint(EndpointName=name, EndpointConfigName=name)
    wait(name, name)
    smoke(name, s['smoke'])
    s['staging_passed'] = True; save(s)


def monitoring(name):
    cw = client('cloudwatch')
    dimensions = [{'Name': 'EndpointName', 'Value': name}, {'Name': 'VariantName', 'Value': 'AllTraffic'}]
    alarm = name + '-5xx'
    cw.put_metric_alarm(AlarmName=alarm, Namespace='AWS/SageMaker', MetricName='Model5XXErrors',
        Dimensions=dimensions, Statistic='Sum', Period=60, EvaluationPeriods=1,
        Threshold=0, ComparisonOperator='GreaterThanThreshold', TreatMissingData='notBreaching')
    metrics = [['AWS/SageMaker', metric, 'EndpointName', name, 'VariantName', 'AllTraffic', {'stat': stat}]
               for metric, stat in [('Invocations', 'Sum'), ('Model5XXErrors', 'Sum'), ('ModelLatency', 'p95')]]
    cw.put_dashboard(DashboardName=name, DashboardBody=json.dumps({'widgets': [{
        'type': 'metric', 'x': 0, 'y': 0, 'width': 24, 'height': 8,
        'properties': {'metrics': metrics, 'region': os.environ['AWS_REGION'],
                       'period': 60, 'title': name}}]}))
    return alarm


def promote():
    s = load()
    if not s.get('staging_passed'):
        raise ValueError('Staging must pass')
    sm = client('sagemaker')
    if sm.describe_model_package(ModelPackageName=s['package_arn'])['ModelApprovalStatus'] != 'Approved':
        raise ValueError('Package is not approved')
    name = os.environ.get('PRODUCTION_ENDPOINT', 'bank-marketing-production')
    current = endpoint_info(name)
    if current and current['EndpointStatus'] != 'InService':
        raise ValueError('Existing endpoint must be InService')
    s['production_endpoint'] = name
    s['previous_config'] = current['EndpointConfigName'] if current else None
    save(s)
    candidate = config(s, 'prod')
    alarm = monitoring(name)
    s['alarm'] = alarm; save(s)
    # Rollback is attempted only after production mutation, never for a CI failure.
    try:
        if current:
            sm.update_endpoint(EndpointName=name, EndpointConfigName=candidate,
                DeploymentConfig={'BlueGreenUpdatePolicy': {
                    'TrafficRoutingConfiguration': {'Type': 'CANARY',
                        'CanarySize': {'Type': 'CAPACITY_PERCENT', 'Value': 50}, 'WaitIntervalInSeconds': 300},
                    'TerminationWaitInSeconds': 300, 'MaximumExecutionTimeoutInSeconds': 3600},
                    'AutoRollbackConfiguration': {'Alarms': [{'AlarmName': alarm}]}})
        else:
            sm.create_endpoint(EndpointName=name, EndpointConfigName=candidate)
        wait(name, candidate)
        smoke(name, s['smoke'])
        s['release_status'] = 'success'; save(s)
    except Exception:
        s['release_status'] = 'failed'; save(s)
        info = endpoint_info(name)
        if current and info and info['EndpointStatus'] == 'InService':
            if info['EndpointConfigName'] != s['previous_config']:
                sm.update_endpoint(EndpointName=name, EndpointConfigName=s['previous_config'])
                wait(name, s['previous_config'])
            s['rollback_status'] = 'restored_previous_config'
        elif not current and info:
            sm.delete_endpoint(EndpointName=name)
            sm.get_waiter('endpoint_deleted').wait(EndpointName=name)
            s['rollback_status'] = 'removed_failed_first_deployment'
        else:
            s['rollback_status'] = 'needs_operator_review_endpoint_not_stable'
        save(s)
        raise


def cleanup():
    """Only delete this run's staging resources. Never delete production here."""
    s = load(); sm = client('sagemaker')
    name = s.get('stage_endpoint')
    if name and endpoint_info(name):
        sm.delete_endpoint(EndpointName=name)
        sm.get_waiter('endpoint_deleted').wait(EndpointName=name,
            WaiterConfig={'Delay': 30, 'MaxAttempts': 180})
    for key, method, arg in [('stage_config', 'delete_endpoint_config', 'EndpointConfigName'),
                             ('stage_model', 'delete_model', 'ModelName')]:
        if s.get(key):
            try:
                getattr(sm, method)(**{arg: s[key]})
            except ClientError as exc:
                if not missing(exc):
                    raise
    s['staging_cleanup'] = 'completed'; save(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'register', 'approve',
                                          'staging', 'promote', 'cleanup'])
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == '__main__':
    main()
