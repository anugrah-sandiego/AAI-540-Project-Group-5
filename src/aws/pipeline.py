"""
SageMaker Pipeline for the scheduled training run.

prepare data -> enough new data? -> train challenger -> evaluate vs champion
-> gates passed? -> register (pending approval) | fail with the reasons

Every step runs the project's own image from ECR, so the cloud trains and scores
with exactly the code and package versions tested locally.
"""

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from src.aws.config import MODEL_GROUP, PIPELINE_NAME, AwsConfig

PROCESSING = '/opt/ml/processing'


def build_pipeline(config: AwsConfig, image_tag: str, session: PipelineSession) -> Pipeline:
    """Define the pipeline for the image pushed with the given tag."""
    image_uri = config.image_uri(image_tag)
    instance_type = ParameterString('InstanceType', default_value='ml.m5.large')
    min_new_rows = ParameterInteger('MinNewRows', default_value=1000)
    quick = ParameterString('QuickTraining', default_value='false')
    run_dir = Join(on='/', values=[config.s3('runs'), ExecutionVariables.PIPELINE_EXECUTION_ID])

    def processor(module: str) -> Processor:
        return Processor(role=config.role_arn, image_uri=image_uri, instance_count=1,
                         instance_type=instance_type, entrypoint=['python', '-m', module],
                         sagemaker_session=session)

    report = PropertyFile(name='PrepareReport', output_name='report', path='report.json')
    prepare = ProcessingStep(
        name='PrepareData',
        step_args=processor('src.aws.steps.prepare').run(
            inputs=[ProcessingInput(source=config.s3('raw'), destination=f'{PROCESSING}/raw'),
                    ProcessingInput(source=config.s3('checkpoints'), destination=f'{PROCESSING}/checkpoint-in')],
            outputs=[ProcessingOutput(output_name=name, source=f'{PROCESSING}/{name}',
                                      destination=Join(on='/', values=[run_dir, name]))
                     for name in ('train', 'eval', 'report')]
            + [ProcessingOutput(output_name='checkpoint', source=f'{PROCESSING}/checkpoint',
                                destination=config.s3('checkpoints')),
               ProcessingOutput(output_name='quarantine', source=f'{PROCESSING}/quarantine',
                                destination=config.s3('quarantine'))],
        ),
        property_files=[report],
    )
    prepared = prepare.properties.ProcessingOutputConfig.Outputs

    estimator = Estimator(image_uri=image_uri, role=config.role_arn, instance_count=1,
                          instance_type=instance_type, output_path=config.s3('models'),
                          hyperparameters={'quick': quick}, sagemaker_session=session)
    train = TrainingStep(
        name='TrainChallenger',
        step_args=estimator.fit({'train': TrainingInput(prepared['train'].S3Output.S3Uri)}),
    )
    model_data = train.properties.ModelArtifacts.S3ModelArtifacts

    evaluation = PropertyFile(name='Evaluation', output_name='evaluation', path='evaluation.json')
    evaluate = ProcessingStep(
        name='EvaluateChallenger',
        step_args=processor('src.aws.steps.evaluate').run(
            inputs=[ProcessingInput(source=model_data, destination=f'{PROCESSING}/challenger'),
                    ProcessingInput(source=prepared['eval'].S3Output.S3Uri, destination=f'{PROCESSING}/eval')],
            outputs=[ProcessingOutput(output_name='evaluation', source=f'{PROCESSING}/evaluation',
                                      destination=Join(on='/', values=[run_dir, 'evaluation']))],
            arguments=['--model-group', MODEL_GROUP, '--region', config.region],
        ),
        property_files=[evaluation],
    )

    model = Model(image_uri=image_uri, model_data=model_data, role=config.role_arn, sagemaker_session=session)
    evaluation_uri = Join(on='/', values=[run_dir, 'evaluation', 'evaluation.json'])
    register = ModelStep(
        name='RegisterChallenger',
        step_args=model.register(
            content_types=['application/json'], response_types=['application/json'],
            inference_instances=['ml.m5.large'], transform_instances=['ml.m5.large'],
            model_package_group_name=MODEL_GROUP, approval_status='PendingManualApproval',
            model_metrics=ModelMetrics(model_statistics=MetricsSource(
                s3_uri=evaluation_uri, content_type='application/json')),
        ),
    )
    reject = FailStep(name='RejectChallenger',
                      error_message=Join(on=' ', values=['Challenger rejected, see', evaluation_uri]))

    gates = ConditionStep(
        name='GatesPassed',
        conditions=[ConditionEquals(left=JsonGet(step_name=evaluate.name, property_file=evaluation,
                                                 json_path='passed'), right=True)],
        if_steps=[register], else_steps=[reject],
    )
    enough_data = ConditionStep(
        name='EnoughNewData',
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=prepare.name, property_file=report, json_path='eval_rows'),
            right=min_new_rows)],
        if_steps=[train, evaluate, gates], else_steps=[],
    )
    return Pipeline(name=PIPELINE_NAME, parameters=[instance_type, min_new_rows, quick],
                    steps=[prepare, enough_data], sagemaker_session=session)
