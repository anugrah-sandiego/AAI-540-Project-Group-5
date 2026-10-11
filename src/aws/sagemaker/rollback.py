"""
Rollback utilities for SageMaker deployment failures.
"""

import os
import boto3
import sagemaker
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerRollback:
    """Manage rollback operations for SageMaker deployments."""

    def __init__(self):
        """Initialize SageMaker session and configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')

        self.sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        self.sm_client = boto3.client('sagemaker', region_name=self.region)

        logger.info(f"Initialized rollback utilities for region: {self.region}")

    def rollback_endpoint(self, endpoint_name: str, previous_model_package_arn: str):
        """
        Rollback an endpoint to a previous model version.

        Args:
            endpoint_name: Name of the endpoint to rollback
            previous_model_package_arn: ARN of the previous model package
        """
        logger.info(f"Rolling back endpoint {endpoint_name} to {previous_model_package_arn}")

        # Get current endpoint config
        endpoint_config = self.sm_client.describe_endpoint_config(
            EndpointConfigName=f"{endpoint_name}-config"
        )

        # Create new endpoint config with previous model
        new_config_name = f"{endpoint_name}-config-rollback-{int(time.time())}"
        self.sm_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': previous_model_package_arn,
                'InitialInstanceCount': endpoint_config['ProductionVariants'][0]['InitialInstanceCount'],
                'InstanceType': endpoint_config['ProductionVariants'][0]['InstanceType']
            }]
        )

        # Update endpoint to use new config
        self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name
        )

        logger.info(f"Endpoint {endpoint_name} rolled back successfully")

    def list_endpoints(self) -> list:
        """List all SageMaker endpoints."""
        response = self.sm_client.list_endpoints()
        return [ep['EndpointName'] for ep in response['Endpoints']]

    def get_latest_approved_model(self, model_package_group_name: str) -> str:
        """
        Get the latest approved model package ARN.

        Args:
            model_package_group_name: Name of the model package group

        Returns:
            ARN of the latest approved model
        """
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if not response['ModelPackageSummaryList']:
            raise ValueError("No approved models found")

        return response['ModelPackageSummaryList'][0]['ModelPackageArn']


def main():
    """Main function to perform rollback."""
    rollback = SageMakerRollback()

    # Example rollback
    # endpoint_name = "bank-marketing-endpoint"
    # model_package_group_name = "bank-marketing-models"
    #
    # previous_model = rollback.get_latest_approved_model(model_package_group_name)
    # rollback.rollback_endpoint(endpoint_name, previous_model)

    print("Rollback utilities initialized")


if __name__ == "__main__":
    main()
