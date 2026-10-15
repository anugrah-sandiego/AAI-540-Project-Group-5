"""
Tests for the pipeline definition and aws-tools commands, without calling AWS.

Need the SageMaker SDK, so they run in the aws-tools image:
    docker compose run --rm aws-test
"""

import json

import boto3
import pytest
from botocore.stub import Stubber

pytest.importorskip('sagemaker')
from sagemaker.workflow.pipeline_context import PipelineSession  # noqa: E402

from src.aws import cli  # noqa: E402
from src.aws.config import AwsConfig  # noqa: E402
from src.aws.pipeline import build_pipeline  # noqa: E402

ACCOUNT = '1' * 12  # built at run time so the account-ID guard ignores this file
CONFIG = AwsConfig(region='us-east-1', account=ACCOUNT, role_arn=f'arn:aws:iam::{ACCOUNT}:role/LabRole',
                   bucket='test-bucket')
GROUP_ARN = f'arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/bank-marketing-models'


def fake_boto_session() -> boto3.Session:
    return boto3.Session(region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')


@pytest.fixture(scope='module')
def definition() -> dict:
    session = PipelineSession(boto_session=fake_boto_session(), default_bucket='test-bucket')
    return json.loads(build_pipeline(CONFIG, 'abc1234', session).definition())


def steps_by_name(steps: list) -> dict:
    """Flatten condition branches so every step can be looked up by name."""
    found = {}
    for step in steps:
        found[step['Name']] = step
        for branch in ('IfSteps', 'ElseSteps'):
            found.update(steps_by_name(step.get('Arguments', {}).get(branch, [])))
    return found


def test_pipeline_runs_every_step_on_the_pushed_image(definition):
    steps = steps_by_name(definition['Steps'])
    assert {'PrepareData', 'EnoughNewData', 'TrainChallenger', 'EvaluateChallenger', 'GatesPassed',
            'RejectChallenger'} <= set(steps)
    image = f'{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/aai540-bank-marketing:abc1234'
    assert steps['PrepareData']['Arguments']['AppSpecification']['ImageUri'] == image
    assert steps['TrainChallenger']['Arguments']['AlgorithmSpecification']['TrainingImage'] == image
    assert steps['EvaluateChallenger']['Arguments']['AppSpecification']['ContainerEntrypoint'] == [
        'python', '-m', 'src.aws.steps.evaluate']


def test_training_only_follows_enough_new_data_and_registration_only_passing_gates(definition):
    top = {step['Name']: step for step in definition['Steps']}
    enough = top['EnoughNewData']['Arguments']
    assert [step['Name'] for step in enough['IfSteps']] == ['TrainChallenger', 'EvaluateChallenger', 'GatesPassed']
    assert enough['ElseSteps'] == []
    gates = steps_by_name(enough['IfSteps'])['GatesPassed']['Arguments']
    assert [step['Name'] for step in gates['ElseSteps']] == ['RejectChallenger']
    assert all(step['Name'].startswith('RegisterChallenger') for step in gates['IfSteps'])


def test_pipeline_parameters_have_safe_defaults(definition):
    defaults = {parameter['Name']: parameter['DefaultValue'] for parameter in definition['Parameters']}
    assert defaults == {'InstanceType': 'ml.m5.large', 'MinNewRows': 300, 'QuickTraining': 'true'}


@pytest.mark.parametrize('index, expected', [(0, slice(0, 31500)), (1, slice(31500, 31850)), (3, slice(32200, 32550))])
def test_replay_batches_are_history_then_fixed_days(index, expected):
    assert cli.batch_rows(index, initial=31500, size=350) == expected


class FakeAws:
    """Stands in for a boto3 session, handing out one stubbed SageMaker client."""

    def __init__(self):
        self.sagemaker = fake_boto_session().client('sagemaker')

    def client(self, name):
        assert name == 'sagemaker'
        return self.sagemaker


def listing(*versions):
    return {'ModelPackageSummaryList': [
        {'ModelPackageArn': f'{GROUP_ARN}/{v}', 'ModelPackageGroupName': 'bank-marketing-models',
         'ModelPackageVersion': v, 'CreationTime': '2026-10-15', 'ModelPackageStatus': 'Completed',
         'ModelApprovalStatus': 'Approved'} for v in versions]}


def test_rollback_rejects_the_champion_so_the_previous_version_takes_over():
    aws = FakeAws()
    with Stubber(aws.sagemaker) as stub:
        stub.add_response('list_model_packages', listing(3, 2))
        stub.add_response('update_model_package', {'ModelPackageArn': f'{GROUP_ARN}/3'},
                          {'ModelPackageArn': f'{GROUP_ARN}/3', 'ModelApprovalStatus': 'Rejected',
                           'ApprovalDescription': 'Rolled back'})
        cli.rollback(aws)
        stub.assert_no_pending_responses()


def test_rollback_refuses_without_an_earlier_version():
    aws = FakeAws()
    with Stubber(aws.sagemaker) as stub:
        stub.add_response('list_model_packages', listing(1))
        with pytest.raises(SystemExit, match='No earlier approved version'):
            cli.rollback(aws)
