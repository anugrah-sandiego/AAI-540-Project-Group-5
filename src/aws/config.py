"""
AWS settings for aws-tools and the pipeline definition.

Nothing account-specific is hard-coded: the account comes from the credentials in
use, and the role, bucket and region can be overridden from the environment (.env).
"""

import os
from dataclasses import dataclass

import boto3

PREFIX = 'aai540/bank-marketing'
PIPELINE_NAME = 'bank-marketing-daily'
MODEL_GROUP = 'bank-marketing-models'
ECR_REPOSITORY = 'aai540-bank-marketing'


@dataclass(frozen=True)
class AwsConfig:
    region: str
    account: str
    role_arn: str
    bucket: str

    @classmethod
    def from_env(cls, session: boto3.Session) -> 'AwsConfig':
        account = session.client('sts').get_caller_identity()['Account']
        region = session.region_name
        return cls(
            region=region,
            account=account,
            # AWS Academy labs provide LabRole; other accounts set AWS_ROLE_ARN
            role_arn=os.getenv('AWS_ROLE_ARN') or f'arn:aws:iam::{account}:role/LabRole',
            bucket=os.getenv('AWS_BUCKET') or f'sagemaker-{region}-{account}',
        )

    def s3(self, *parts: str) -> str:
        """S3 URI under the project prefix, e.g. s3('raw') -> s3://bucket/aai540/bank-marketing/raw."""
        return '/'.join([f's3://{self.bucket}', PREFIX, *parts])

    def key(self, *parts: str) -> str:
        """Object key under the project prefix."""
        return '/'.join([PREFIX, *parts])

    def image_uri(self, tag: str) -> str:
        return f'{self.account}.dkr.ecr.{self.region}.amazonaws.com/{ECR_REPOSITORY}:{tag}'


def session() -> boto3.Session:
    return boto3.Session(region_name=os.getenv('AWS_REGION', 'us-east-1'))
