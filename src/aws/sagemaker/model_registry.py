"""
SageMaker Model Registry integration for model versioning and management.
"""

import os
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.model_registry import ModelPackageGroup, ModelPackage
from dotenv import load_dotenv
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerModelRegistry:
    """Manage SageMaker Model Registry for model versioning."""

    def __init__(self):
        """Initialize SageMaker session and configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.bucket_name = f"sagemaker-{self.region}-{self.account_id}"
        self.base_prefix = "aai540/bank-marketing"

        self.sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        self.sm_client = boto3.client('sagemaker', region_name=self.region)

        # Model package group name
        self.model_package_group_name = "bank-marketing-models"

        logger.info(f"Initialized Model Registry for region: {self.region}")

    def create_model_package_group(self, description: str = "Bank Marketing Model Package Group"):
        """
        Create a Model Package Group if it doesn't exist.

        Args:
            description: Description of the model package group

        Returns:
            ModelPackageGroup object
        """
        try:
            # Check if group already exists
            response = self.sm_client.describe_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
            logger.info(f"Model package group {self.model_package_group_name} already exists")
            return ModelPackageGroup(
                name=self.model_package_group_name,
                sagemaker_session=self.sagemaker_session
            )
        except self.sm_client.exceptions.ResourceNotFound:
            pass

        # Create new model package group
        logger.info(f"Creating model package group: {self.model_package_group_name}")
        model_package_group = ModelPackageGroup(
            name=self.model_package_group_name,
            description=description,
            sagemaker_session=self.sagemaker_session
        )

        model_package_group.create()
        logger.info(f"Model package group {self.model_package_group_name} created")
        return model_package_group

    def register_model(self, model_s3_uri: str, model_metrics: dict = None,
                      model_description: str = "Random Forest model for term deposit prediction",
                      approval_status: str = "PendingManualApproval"):
        """
        Register a trained model in the Model Registry.

        Args:
            model_s3_uri: S3 URI of the trained model
            model_metrics: Dictionary of model metrics
            model_description: Description of the model
            approval_status: Initial approval status

        Returns:
            ModelPackage object
        """
        # Ensure model package group exists
        model_package_group = self.create_model_package_group()

        # Create model package
        model_package = ModelPackageGroup.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge", "ml.m5.large"],
            model_package_group_name=self.model_package_group_name,
            model_package_description=model_description,
            model_metrics=model_metrics,
            approval_status=approval_status,
            source_uri=model_s3_uri,
            image_uri=f"683313688378.dkr.ecr.{self.region}.amazonaws.com/sagemaker-scikit-learn:1.2-1",
            sagemaker_session=self.sagemaker_session
        )

        logger.info(f"Model registered with package: {model_package.model_package_arn}")
        return model_package

    def list_model_packages(self, status: str = None) -> list:
        """
        List all model packages in the group.

        Args:
            status: Filter by status (Approved, Rejected, PendingManualApproval)

        Returns:
            List of model package information
        """
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=self.model_package_group_name,
            ModelApprovalStatus=status
        )

        packages = []
        for pkg in response['ModelPackageSummaryList']:
            packages.append({
                'model_package_arn': pkg['ModelPackageArn'],
                'model_package_name': pkg['ModelPackageName'],
                'version': pkg['ModelPackageVersion'],
                'approval_status': pkg['ModelApprovalStatus'],
                'creation_time': pkg['CreationTime']
            })

        return packages

    def get_latest_approved_model(self) -> dict:
        """
        Get the latest approved model package.

        Returns:
            Dictionary with model package information
        """
        packages = self.list_model_packages(status='Approved')

        if not packages:
            raise ValueError("No approved models found in the registry")

        # Sort by creation time and get the latest
        latest = max(packages, key=lambda x: x['creation_time'])
        logger.info(f"Latest approved model: {latest['model_package_arn']}")
        return latest

    def update_model_status(self, model_package_arn: str,
                            approval_status: str,
                            approval_description: str = None):
        """
        Update the approval status of a model package.

        Args:
            model_package_arn: ARN of the model package
            approval_status: New approval status (Approved, Rejected, PendingManualApproval)
            approval_description: Description of the approval decision
        """
        logger.info(f"Updating model {model_package_arn} status to {approval_status}")
        self.sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus=approval_status,
            ApprovalDescription=approval_description
        )
        logger.info(f"Model status updated")

    def delete_model_package(self, model_package_arn: str):
        """
        Delete a model package from the registry.

        Args:
            model_package_arn: ARN of the model package to delete
        """
        logger.info(f"Deleting model package: {model_package_arn}")
        self.sm_client.delete_model_package(ModelPackageArn=model_package_arn)
        logger.info(f"Model package deleted")

    def describe_model_package(self, model_package_arn: str) -> dict:
        """
        Get detailed information about a model package.

        Args:
            model_package_arn: ARN of the model package

        Returns:
            Dictionary with model package details
        """
        response = self.sm_client.describe_model_package(ModelPackageArn=model_package_arn)

        return {
            'model_package_arn': response['ModelPackageArn'],
            'model_package_name': response['ModelPackageName'],
            'version': response['ModelPackageVersion'],
            'approval_status': response['ModelApprovalStatus'],
            'creation_time': response['CreationTime'],
            'inference_spec': response.get('InferenceSpecification', {}),
            'model_metrics': response.get('ModelMetrics', {})
        }


def main():
    """Main function to demonstrate model registry operations."""
    registry = SageMakerModelRegistry()

    # Create model package group
    model_package_group = registry.create_model_package_group()

    print(f"Model package group: {model_package_group.name}")

    # Example: Register a model
    # model_s3_uri = f"s3://{registry.bucket_name}/{registry.base_prefix}/models/model.tar.gz"
    # model_metrics = {
    #     'EvaluationMetrics': {
    #         'MetricGroups': [{
    #             'MetricData': [
    #                 {'MetricName': 'accuracy', 'Value': 0.85},
    #                 {'MetricName': 'f1', 'Value': 0.82},
    #                 {'MetricName': 'roc_auc', 'Value': 0.91}
    #             ]
    #         }]
    #     }
    # }
    # model_package = registry.register_model(model_s3_uri, model_metrics)

    # List all models
    packages = registry.list_model_packages()
    print(f"\nModel packages in registry: {len(packages)}")
    for pkg in packages:
        print(f"  - Version {pkg['version']}: {pkg['approval_status']}")


if __name__ == "__main__":
    main()
