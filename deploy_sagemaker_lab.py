"""
SageMaker Lab Environment Deployment Script.

This script is designed for SageMaker lab environments where credentials
are automatically handled. It performs the key AWS SageMaker deployment steps.
"""

import os
import sys
import boto3
import sagemaker
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main deployment function for SageMaker lab environment."""
    logger.info("=" * 60)
    logger.info("SageMaker Lab Environment Deployment")
    logger.info("=" * 60)
    logger.info("")

    # Load environment variables
    load_dotenv()

    # Get configuration
    region = os.getenv('AWS_REGION', 'us-east-1')
    account_id = os.getenv('AWS_ACCOUNT_ID', '562792440372')
    role_arn = os.getenv('AWS_ROLE_ARN', f'arn:aws:iam::{account_id}:role/LabRole')

    logger.info(f"Region: {region}")
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Role ARN: {role_arn}")
    logger.info("")

    # Initialize SageMaker session (lab environment handles credentials automatically)
    try:
        sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=region))
        logger.info("✓ SageMaker session initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize SageMaker session: {e}")
        return False

    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=region)
        bucket_name = f"sagemaker-{region}-{account_id}"
        logger.info(f"✓ S3 client initialized")
        logger.info(f"  Bucket: {bucket_name}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize S3 client: {e}")
        return False

    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"✓ S3 bucket {bucket_name} exists")
    except:
        logger.warning(f"⚠ S3 bucket {bucket_name} does not exist yet")
        logger.info("  It will be created automatically when needed")

    # Initialize SageMaker client
    try:
        sm_client = boto3.client('sagemaker', region_name=region)
        logger.info("✓ SageMaker client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize SageMaker client: {e}")
        return False

    # Initialize CloudWatch client
    try:
        cw_client = boto3.client('cloudwatch', region_name=region)
        logger.info("✓ CloudWatch client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize CloudWatch client: {e}")
        return False

    logger.info("")
    logger.info("=" * 60)
    logger.info("Environment Validation Complete")
    logger.info("=" * 60)
    logger.info("")
    logger.info("✓ All AWS clients initialized successfully")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("1. Upload data to S3:")
    logger.info("   python -m src.aws.sagemaker.data_upload")
    logger.info("")
    logger.info("2. Create Feature Store:")
    logger.info("   python -m src.aws.sagemaker.feature_store")
    logger.info("")
    logger.info("3. Run training:")
    logger.info("   python -m src.aws.sagemaker.training")
    logger.info("")
    logger.info("4. Register model:")
    logger.info("   python -m src.aws.sagemaker.model_registry")
    logger.info("")
    logger.info("5. Deploy endpoint:")
    logger.info("   python -m src.aws.sagemaker.deployment")
    logger.info("")
    logger.info("6. Set up monitoring:")
    logger.info("   python -m src.aws.sagemaker.monitoring")
    logger.info("")
    logger.info("7. Create dashboards:")
    logger.info("   python -m src.aws.sagemaker.infrastructure_monitoring")
    logger.info("")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
