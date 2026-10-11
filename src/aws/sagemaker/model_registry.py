"""Register SageMaker training artifacts using supported boto3 APIs."""
import logging
import os
import boto3
import sagemaker
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.model import ModelPackage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerModelRegistry:
    def __init__(self):
        load_dotenv()
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        self.sm_client = self.sagemaker_session.sagemaker_client
        self.model_package_group_name = 'bank-marketing-models'

    def create_model_package_group(self, description='Bank Marketing Model Package Group'):
        try:
            return self.sm_client.describe_model_package_group(
                ModelPackageGroupName=self.model_package_group_name)
        except ClientError as exc:
            error = exc.response['Error']
            message = error.get('Message', '').lower()
            missing = error['Code'] in ('ResourceNotFound', 'ResourceNotFoundException') or (
                error['Code'] == 'ValidationException' and
                any(term in message for term in ('does not exist', 'not found', 'could not find')))
            if not missing:
                raise
        return self.sm_client.create_model_package_group(
            ModelPackageGroupName=self.model_package_group_name,
            ModelPackageGroupDescription=description)

    def register_model(self, model_s3_uri, model_metrics=None,
                       model_description='Random Forest model for term deposit prediction',
                       approval_status='PendingManualApproval', image_uri=None):
        if not image_uri:
            raise ValueError('Provide the verified inference image URI from the training job.')
        self.create_model_package_group()
        request = dict(
            ModelPackageGroupName=self.model_package_group_name,
            ModelPackageDescription=model_description,
            ModelApprovalStatus=approval_status,
            InferenceSpecification={
                'Containers': [{'Image': image_uri, 'ModelDataUrl': model_s3_uri}],
                'SupportedContentTypes': ['text/csv'],
                'SupportedResponseMIMETypes': ['text/csv'],
                'SupportedRealtimeInferenceInstanceTypes': ['ml.m5.xlarge'],
                'SupportedTransformInstanceTypes': ['ml.m5.xlarge'],
            })
        if model_metrics is not None:
            request['ModelMetrics'] = model_metrics
        response = self.sm_client.create_model_package(**request)
        arn = response['ModelPackageArn']
        logger.info('Registered model package: %s', arn)
        return ModelPackage(role=self.role_arn, model_package_arn=arn,
                            sagemaker_session=self.sagemaker_session)

    def list_model_packages(self, status=None):
        kwargs = {'ModelPackageGroupName': self.model_package_group_name}
        if status:
            kwargs['ModelApprovalStatus'] = status
        packages = []
        for page in self.sm_client.get_paginator('list_model_packages').paginate(**kwargs):
            for item in page['ModelPackageSummaryList']:
                packages.append(dict(model_package_arn=item['ModelPackageArn'],
                                     model_package_name=item.get('ModelPackageName'),
                                     version=item.get('ModelPackageVersion'),
                                     approval_status=item['ModelApprovalStatus'],
                                     creation_time=item['CreationTime']))
        return packages

    def get_latest_approved_model(self):
        packages = self.list_model_packages('Approved')
        if not packages:
            raise ValueError('No approved models found in the registry')
        return max(packages, key=lambda item: item['creation_time'])

    def update_model_status(self, model_package_arn, approval_status, approval_description=None):
        kwargs = dict(ModelPackageArn=model_package_arn, ModelApprovalStatus=approval_status)
        if approval_description is not None:
            kwargs['ApprovalDescription'] = approval_description
        return self.sm_client.update_model_package(**kwargs)

    def delete_model_package(self, model_package_arn):
        return self.sm_client.delete_model_package(ModelPackageArn=model_package_arn)

    def describe_model_package(self, model_package_arn):
        item = self.sm_client.describe_model_package(ModelPackageArn=model_package_arn)
        return dict(model_package_arn=item['ModelPackageArn'],
                    model_package_name=item.get('ModelPackageName'),
                    version=item.get('ModelPackageVersion'),
                    approval_status=item['ModelApprovalStatus'],
                    creation_time=item['CreationTime'],
                    inference_spec=item.get('InferenceSpecification', {}),
                    model_metrics=item.get('ModelMetrics', {}))


def main():
    registry = SageMakerModelRegistry()
    job_name = os.getenv('TRAINING_JOB_NAME')
    if not job_name:
        raise ValueError('Set TRAINING_JOB_NAME to the completed training job to register.')
    job = registry.sm_client.describe_training_job(TrainingJobName=job_name)
    if job['TrainingJobStatus'] != 'Completed':
        raise ValueError(f'Training job is not completed: {job["TrainingJobStatus"]}')
    package = registry.register_model(
        model_s3_uri=job['ModelArtifacts']['S3ModelArtifacts'],
        image_uri=job['AlgorithmSpecification']['TrainingImage'],
        model_description=f'Random Forest artifact from {job_name}; inference packaging requires validation.')
    print(f'Model package ARN: {package.model_package_arn}')
    print('Approval status: PendingManualApproval')


if __name__ == '__main__':
    main()
