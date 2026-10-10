"""
AWS SageMaker setup validation script.

This script validates that all AWS credentials and permissions are correctly configured
before attempting to deploy to SageMaker.
"""

import os
import sys
import boto3
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_env_file():
    """Validate that .env file exists and contains required variables."""
    logger.info("Validating .env file...")

    if not os.path.exists('.env'):
        logger.error(".env file not found. Please create it with your AWS credentials.")
        return False

    load_dotenv()

    required_vars = ['AWS_ACCOUNT_ID', 'AWS_REGION', 'AWS_ROLE_ARN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    logger.info(f".env file validated successfully")
    logger.info(f"  Account ID: {os.getenv('AWS_ACCOUNT_ID')}")
    logger.info(f"  Region: {os.getenv('AWS_REGION')}")
    logger.info(f"  Role ARN: {os.getenv('AWS_ROLE_ARN')}")

    return True


def validate_aws_credentials():
    """Validate AWS credentials are correctly configured."""
    logger.info("Validating AWS credentials...")

    try:
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials is None:
            logger.error("AWS credentials not found. Please configure them.")
            return False

        logger.info("AWS credentials found")
        return True

    except Exception as e:
        logger.error(f"Error validating AWS credentials: {e}")
        return False


def validate_s3_access():
    """Validate S3 access and bucket creation."""
    logger.info("Validating S3 access...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        account_id = os.getenv('AWS_ACCOUNT_ID')
        bucket_name = f"sagemaker-{region}-{account_id}"

        s3_client = boto3.client('s3', region_name=region)

        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket {bucket_name} exists and is accessible")
        except:
            logger.warning(f"S3 bucket {bucket_name} does not exist. It will be created automatically.")
            logger.info(f"Expected bucket name: {bucket_name}")

        return True

    except Exception as e:
        logger.error(f"Error validating S3 access: {e}")
        return False


def validate_sagemaker_role():
    """Validate SageMaker execution role exists and is accessible."""
    logger.info("Validating SageMaker role...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        role_arn = os.getenv('AWS_ROLE_ARN')

        iam_client = boto3.client('iam', region_name=region)

        # Extract role name from ARN
        role_name = role_arn.split('/')[-1]

        try:
            iam_client.get_role(RoleName=role_name)
            logger.info(f"SageMaker role {role_name} exists and is accessible")
            return True
        except iam_client.exceptions.NoSuchEntityException:
            logger.error(f"SageMaker role {role_name} does not exist")
            return False

    except Exception as e:
        logger.error(f"Error validating SageMaker role: {e}")
        return False


def validate_sagemaker_permissions():
    """Validate that the role has necessary SageMaker permissions."""
    logger.info("Validating SageMaker permissions...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        sagemaker_client = boto3.client('sagemaker', region_name=region)

        # Try to list training jobs (should work even if empty)
        sagemaker_client.list_training_jobs(MaxResults=1)

        logger.info("SageMaker permissions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating SageMaker permissions: {e}")
        logger.error("Ensure your IAM role has SageMakerFullAccess or equivalent permissions")
        return False


def validate_cloudwatch_permissions():
    """Validate CloudWatch permissions for monitoring."""
    logger.info("Validating CloudWatch permissions...")

    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        cloudwatch = boto3.client('cloudwatch', region_name=region)

        # Try to list dashboards (should work even if empty)
        cloudwatch.list_dashboards()

        logger.info("CloudWatch permissions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating CloudWatch permissions: {e}")
        logger.error("Ensure your IAM role has CloudWatchFullAccess or equivalent permissions")
        return False


def validate_python_dependencies():
    """Validate that required Python packages are installed."""
    logger.info("Validating Python dependencies...")

    required_packages = [
        'boto3',
        'sagemaker',
        'pandas',
        'numpy',
        'scikit-learn',
        'mlflow'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    logger.info("All required Python packages are installed")
    return True


def validate_data_files():
    """Validate that required data files exist."""
    logger.info("Validating data files...")

    required_files = [
        'data/raw/bank-full.csv',
        'data/raw/bank.csv',
        'data/raw/bank-names.txt'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        return False

    logger.info("All required data files exist")
    return True


def run_validation():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("AWS SageMaker Setup Validation")
    logger.info("=" * 60)
    logger.info("")

    checks = [
        ("Environment File", validate_env_file),
        ("AWS Credentials", validate_aws_credentials),
        ("S3 Access", validate_s3_access),
        ("SageMaker Role", validate_sagemaker_role),
        ("SageMaker Permissions", validate_sagemaker_permissions),
        ("CloudWatch Permissions", validate_cloudwatch_permissions),
        ("Python Dependencies", validate_python_dependencies),
        ("Data Files", validate_data_files),
    ]

    results = []

    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        result = check_func()
        results.append((check_name, result))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")

    logger.info("")
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("All validations passed! You can proceed with deployment.")
        return True
    else:
        logger.error("Some validations failed. Please fix the issues above before deploying.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
