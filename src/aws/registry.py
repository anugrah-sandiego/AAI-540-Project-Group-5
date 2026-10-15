"""
Model Registry lookups shared by the pipeline's evaluation step and aws-tools.

The champion is the most recently approved version in the model package group;
rejecting it makes the previous approved version the champion again.
"""

from typing import List


def latest_packages(sagemaker, group: str, status: str, limit: int = 1) -> List[str]:
    """ARNs of the newest model packages with the given approval status, newest first."""
    response = sagemaker.list_model_packages(
        ModelPackageGroupName=group, ModelApprovalStatus=status,
        SortBy='CreationTime', SortOrder='Descending', MaxResults=limit,
    )
    return [package['ModelPackageArn'] for package in response['ModelPackageSummaryList']]


def model_data_url(sagemaker, package_arn: str) -> str:
    """S3 URI of the model.tar.gz a package was registered with."""
    package = sagemaker.describe_model_package(ModelPackageName=package_arn)
    return package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
