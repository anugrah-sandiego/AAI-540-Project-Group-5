"""
SageMaker batch transform job implementation for batch inference.
"""

import os
import boto3
import time
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.transformer import Transformer
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerBatchInference:
    """Manage SageMaker batch transform jobs for batch inference."""

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

        logger.info(f"Initialized SageMaker batch inference for region: {self.region}")

    def prepare_batch_transform_script(self):
        """
        Prepare the batch transform script for SageMaker.

        This script handles batch prediction requests.
        """
        transform_script_content = '''
import json
import joblib
import os
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Load the model from model_dir."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == 'text/csv':
        # Parse CSV input (multiple rows)
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            if line:
                data.append([float(x) for x in line.split(',')])
        return np.array(data)
    elif request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    # Get probability predictions
    proba = model.predict_proba(input_data)[:, 1]
    return proba


def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'text/csv':
        # Return CSV format
        lines = [','.join([str(x) for x in row]) for row in prediction]
        return '\n'.join(lines)
    elif accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''

        # Save transform script
        script_dir = Path("src/aws/sagemaker/scripts")
        script_dir.mkdir(parents=True, exist_ok=True)

        script_path = script_dir / "batch_transform.py"
        with open(script_path, 'w') as f:
            f.write(transform_script_content)

        logger.info(f"Batch transform script saved to {script_path}")
        return str(script_path)

    def create_transformer(self, model_s3_uri: str,
                          instance_type: str = 'ml.m5.xlarge',
                          instance_count: int = 1):
        """
        Create a Transformer object for batch inference.

        Args:
            model_s3_uri: S3 URI of the trained model
            instance_type: Instance type for the transform job
            instance_count: Number of instances

        Returns:
            Transformer object
        """
        # Prepare transform script
        script_path = self.prepare_batch_transform_script()

        # Create SKLearn model
        model = SKLearnModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            entry_point='batch_transform.py',
            source_dir=str(Path(script_path).parent),
            framework_version='1.2-1',
            py_version='py3',
            sagemaker_session=self.sagemaker_session
        )

        # Create transformer
        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=f"s3://{self.bucket_name}/{self.base_prefix}/inference/batch-output/",
            sagemaker_session=self.sagemaker_session
        )

        logger.info("Transformer created")
        return transformer

    def run_batch_transform(self, transformer: Transformer,
                           input_s3_uri: str,
                           job_name: str = None,
                           content_type: str = 'text/csv',
                           split_type: str = 'Line'):
        """
        Run a batch transform job.

        Args:
            transformer: Transformer object
            input_s3_uri: S3 URI of input data
            job_name: Name for the transform job
            content_type: Content type of input data
            split_type: How to split input data (Line, Record, etc.)

        Returns:
            Transform job name
        """
        if job_name is None:
            job_name = f"bank-marketing-batch-transform-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        logger.info(f"Starting batch transform job: {job_name}")
        transformer.transform(
            data=input_s3_uri,
            content_type=content_type,
            split_type=split_type,
            job_name=job_name,
            wait=False
        )

        logger.info(f"Batch transform job {job_name} started")
        return job_name

    def get_transform_job_status(self, job_name: str) -> dict:
        """
        Get the status of a transform job.

        Args:
            job_name: Name of the transform job

        Returns:
            Dictionary with job status information
        """
        response = self.sm_client.describe_transform_job(TransformJobName=job_name)
        return {
            'status': response['TransformJobStatus'],
            'start_time': response.get('TransformStartTime'),
            'end_time': response.get('TransformEndTime'),
            'instance_type': response.get('TransformResources', {}).get('InstanceType'),
            'instance_count': response.get('TransformResources', {}).get('InstanceCount')
        }

    def wait_for_transform_job(self, job_name: str, poll_interval: int = 30):
        """
        Wait for a transform job to complete.

        Args:
            job_name: Name of the transform job
            poll_interval: Polling interval in seconds
        """
        logger.info(f"Waiting for transform job {job_name} to complete...")

        while True:
            status_info = self.get_transform_job_status(job_name)
            status = status_info['status']

            logger.info(f"Transform job status: {status}")

            if status in ['Completed', 'Failed', 'Stopped']:
                break

            time.sleep(poll_interval)

        logger.info(f"Transform job {job_name} finished with status: {status}")
        return status_info

    def list_transform_jobs(self) -> list:
        """
        List all transform jobs.

        Returns:
            List of transform job names
        """
        response = self.sm_client.list_transform_jobs()
        return [job['TransformJobName'] for job in response['TransformJobSummaries']]


def main():
    """Main function to run batch inference."""
    batch_inference = SageMakerBatchInference()

    # Example model S3 URI (would be obtained from training job)
    model_s3_uri = f"s3://{batch_inference.bucket_name}/{batch_inference.base_prefix}/models/model.tar.gz"

    # Example input S3 URI
    input_s3_uri = f"s3://{batch_inference.bucket_name}/{batch_inference.base_prefix}/inference/batch-input/"

    # Create transformer
    transformer = batch_inference.create_transformer(model_s3_uri)

    # Run batch transform
    job_name = batch_inference.run_batch_transform(transformer, input_s3_uri)

    print(f"Batch transform job started: {job_name}")

    # Wait for completion
    status_info = batch_inference.wait_for_transform_job(job_name)
    print(f"Transform job completed: {status_info}")


if __name__ == "__main__":
    main()
