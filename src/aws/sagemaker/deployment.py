"""
SageMaker model deployment and endpoint management.
"""

# CI uses the shared fitted-preprocessing release engine. Legacy library APIs remain below.
import os as _ci_os

if __name__ == "__main__" and _ci_os.environ.get("GROUP5_CI") == "1":
    from src.aws.sagemaker.ci_entrypoints import run as _ci_run
    _ci_run("deployment")
    raise SystemExit(0)


import os
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.serializers import CSVSerializer
from dotenv import load_dotenv
import logging
from datetime import datetime
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployment:
    """Manage SageMaker model deployment and endpoints."""

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

        logger.info(f"Initialized SageMaker deployment for region: {self.region}")

    def prepare_inference_script(self):
        """
        Prepare the inference script for SageMaker endpoint.

        This script handles prediction requests.
        """
        inference_script_content = '''
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
        # Parse CSV input
        data = request_body.split(',')
        return np.array([float(x) for x in data])
    elif request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    # Reshape if needed
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    # Get probability predictions
    proba = model.predict_proba(input_data)[:, 1]
    return proba


def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'text/csv':
        return ','.join([str(x) for x in prediction])
    elif accept == 'application/json':
        return json.dumps({'prediction': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''

        # Save inference script
        script_dir = Path("src/aws/sagemaker/scripts")
        script_dir.mkdir(parents=True, exist_ok=True)

        script_path = script_dir / "inference.py"
        with open(script_path, 'w') as f:
            f.write(inference_script_content)

        logger.info(f"Inference script saved to {script_path}")
        return str(script_path)

    def deploy_model(self, model_s3_uri: str, endpoint_name: str = None,
                    instance_type: str = 'ml.m5.xlarge',
                    initial_instance_count: int = 1,
                    enable_data_capture: bool = True):
        """
        Deploy a trained model to a SageMaker endpoint.

        Args:
            model_s3_uri: S3 URI of the trained model
            endpoint_name: Name for the endpoint
            instance_type: Instance type for the endpoint
            initial_instance_count: Number of instances
            enable_data_capture: Whether to enable data capture for monitoring

        Returns:
            Predictor object for the endpoint
        """
        if endpoint_name is None:
            endpoint_name = f"bank-marketing-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Prepare inference script
        script_path = self.prepare_inference_script()

        # Create SKLearn model
        model = SKLearnModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            entry_point='inference.py',
            source_dir=str(Path(script_path).parent),
            framework_version='1.2-1',
            py_version='py3',
            sagemaker_session=self.sagemaker_session
        )

        # Configure data capture for monitoring
        data_capture_config = None
        if enable_data_capture:
            capture_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/data-capture/"
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=capture_s3_uri,
                capture_options=['REQUEST', 'RESPONSE'],
                csv_content_types=['text/csv'],
                json_content_types=['application/json']
            )

        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            data_capture_config=data_capture_config
        )

        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return predictor

    def invoke_endpoint(self, endpoint_name: str, data: list,
                        content_type: str = 'text/csv'):
        """
        Invoke a SageMaker endpoint for prediction.

        Args:
            endpoint_name: Name of the endpoint
            data: Input data for prediction
            content_type: Content type of the input

        Returns:
            Prediction results
        """
        runtime_client = boto3.client('runtime.sagemaker', region_name=self.region)

        if content_type == 'text/csv':
            payload = ','.join([str(x) for x in data])
        elif content_type == 'application/json':
            import json
            payload = json.dumps(data)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload
        )

        result = response['Body'].read().decode('utf-8')
        return result

    def delete_endpoint(self, endpoint_name: str):
        """
        Delete a SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete
        """
        logger.info(f"Deleting endpoint: {endpoint_name}")
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} deleted")

    def describe_endpoint(self, endpoint_name: str) -> dict:
        """
        Get endpoint information.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Dictionary with endpoint information
        """
        response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        return {
            'endpoint_name': response['EndpointName'],
            'endpoint_status': response['EndpointStatus'],
            'creation_time': response.get('CreationTime'),
            'endpoint_config_name': response.get('EndpointConfigName'),
            'instance_type': response.get('ProductionVariants', [{}])[0].get('InstanceType'),
            'instance_count': response.get('ProductionVariants', [{}])[0].get('InitialInstanceCount')
        }

    def list_endpoints(self) -> list:
        """
        List all SageMaker endpoints.

        Returns:
            List of endpoint names
        """
        response = self.sm_client.list_endpoints()
        return [ep['EndpointName'] for ep in response['Endpoints']]


def main():
    """Main function to deploy model."""
    deployer = SageMakerDeployment()

    # Example model S3 URI (would be obtained from training job)
    model_s3_uri = (
        "s3://sagemaker-us-east-1-562792440372/"
        "aai540/bank-marketing/models/"
        "bank-marketing-training-20261011-095612/output/model.tar.gz"
    )

    # Deploy model
    predictor = deployer.deploy_model(model_s3_uri)

    print(f"Model deployed. Endpoint: {predictor.endpoint_name}")

    # Example invocation
    # result = deployer.invoke_endpoint(predictor.endpoint_name, [30, 1500, 5, ...])


if __name__ == "__main__":
    main()
