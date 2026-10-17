"""Deploy the selected completed SageMaker training artifact.

The legacy classifier expects already encoded numeric features in training order.
This script does not reconstruct or refit a missing categorical encoder.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.sklearn.model import SKLearnModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFERENCE_SCRIPT = r'''
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(Path(model_dir) / "model.joblib")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Expected a classifier with predict_proba; incompatible artifact.")
    return model


def input_fn(request_body, request_content_type):
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    kind = request_content_type.split(";")[0].strip().lower()
    if kind == "text/csv":
        rows = list(csv.reader(io.StringIO(request_body)))
        if not rows or not rows[0]:
            raise ValueError("CSV must contain numeric rows without a header.")
        frame = pd.DataFrame(rows)
    elif kind == "application/json":
        data = json.loads(request_body)
        columns = None
        if isinstance(data, dict):
            columns = data.get("columns")
            data = data.get("instances")
        if not isinstance(data, list) or not data:
            raise ValueError("Use a numeric list, a list of rows, or an instances object.")
        if isinstance(data[0], dict):
            if columns is not None:
                raise ValueError("Omit columns when instances are named records.")
            frame = pd.DataFrame(data)
        else:
            if not isinstance(data[0], list):
                data = [data]
            frame = pd.DataFrame(data, columns=columns)
    else:
        raise ValueError("Supported input types: text/csv and application/json.")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Input must be nonempty with unique feature columns.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Use already encoded numeric features; raw categories are unsupported.") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Features must be finite numeric values.")
    return frame


def predict_fn(frame, model):
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and frame.shape[1] != expected_count:
        raise ValueError(f"Expected {expected_count} encoded features; received {frame.shape[1]}.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if isinstance(frame.columns, pd.RangeIndex):
            frame = frame.copy()
            frame.columns = names
        else:
            if set(frame.columns) != set(names):
                raise ValueError("Named columns must exactly match encoded training feature names.")
            frame = frame.loc[:, names]
    elif not isinstance(frame.columns, pd.RangeIndex):
        raise ValueError("Model has no saved feature names; send positional features in training order.")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The model must have positive class label 1.")
    probabilities = model.predict_proba(frame)[:, classes.index(1)]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned nonfinite probabilities.")
    return probabilities


def output_fn(prediction, accept):
    kind = (accept or "application/json").split(";")[0].strip().lower()
    if kind == "text/csv":
        return "\n".join(str(float(value)) for value in prediction)
    if kind in ("application/json", "*/*"):
        return json.dumps({"prediction": prediction.tolist()})
    raise ValueError("Supported response types: text/csv and application/json.")
'''


class SageMakerDeployment:
    def __init__(self):
        load_dotenv()
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.role_arn = os.getenv("AWS_ROLE_ARN")
        self.account_id = os.getenv("AWS_ACCOUNT_ID")
        if not self.role_arn:
            raise ValueError("Set AWS_ROLE_ARN to your SageMaker execution role ARN.")
        if not self.account_id:
            raise ValueError("Set AWS_ACCOUNT_ID to your AWS account ID.")
        self.bucket_name = os.getenv("AWS_BUCKET") or f"sagemaker-{self.region}-{self.account_id}"
        self.base_prefix = "aai540/bank-marketing"
        aws_session = boto3.session.Session(region_name=self.region)
        self.sagemaker_session = sagemaker.Session(boto_session=aws_session)
        self.sm_client = aws_session.client("sagemaker")
        self.s3_client = aws_session.client("s3")
        self.runtime_client = aws_session.client("sagemaker-runtime")

    def prepare_inference_script(self):
        # Dedicated folder avoids packaging unrelated training scripts/requirements.
        folder = Path(__file__).resolve().parent / "scripts" / "deployment"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "inference.py"
        path.write_text(INFERENCE_SCRIPT, encoding="utf-8")
        return str(path)

    def verify_model_artifact(self, model_s3_uri):
        parsed = urlparse(model_s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("Expected an s3://bucket/key model artifact URI.")
        self.s3_client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))

    def deploy_model(self, model_s3_uri, endpoint_name=None,
                     instance_type="ml.m5.xlarge", initial_instance_count=1,
                     enable_data_capture=True):
        if initial_instance_count < 1:
            raise ValueError("initial_instance_count must be at least 1.")
        self.verify_model_artifact(model_s3_uri)
        if endpoint_name is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            endpoint_name = f"bank-marketing-endpoint-{stamp}-{uuid4().hex[:6]}"
        script = Path(self.prepare_inference_script())
        model = SKLearnModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            entry_point=script.name,
            source_dir=str(script.parent),
            framework_version="1.2-1",
            py_version="py3",
            sagemaker_session=self.sagemaker_session,
        )
        capture = None
        if enable_data_capture:
            capture = DataCaptureConfig(
                enable_capture=True, sampling_percentage=100,
                destination_s3_uri=f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/data-capture/",
                capture_options=["REQUEST", "RESPONSE"],
                csv_content_types=["text/csv"],
                json_content_types=["application/json"],
                sagemaker_session=self.sagemaker_session,
            )
        logger.info("Deploying %s to %s", model_s3_uri, endpoint_name)
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type, endpoint_name=endpoint_name,
            data_capture_config=capture,
        )
        logger.info("Endpoint deployed: %s", predictor.endpoint_name)
        return predictor

    def invoke_endpoint(self, endpoint_name, data, content_type="text/csv"):
        kind = content_type.split(";")[0].strip().lower()
        if kind == "text/csv":
            import csv
            import io
            rows = data if data and isinstance(data[0], (list, tuple)) else [data]
            stream = io.StringIO()
            csv.writer(stream).writerows(rows)
            payload = stream.getvalue()
        elif kind == "application/json":
            payload = json.dumps(data)
        else:
            raise ValueError("Use text/csv or application/json.")
        response = self.runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=kind, Accept=kind,
            Body=payload.encode("utf-8"),
        )
        try:
            return response["Body"].read().decode("utf-8")
        finally:
            response["Body"].close()

    def describe_endpoint(self, endpoint_name):
        endpoint = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        config = self.sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint["EndpointConfigName"]
        )
        live = {v["VariantName"]: v for v in endpoint.get("ProductionVariants", [])}
        return {
            "endpoint_name": endpoint["EndpointName"],
            "endpoint_status": endpoint["EndpointStatus"],
            "creation_time": endpoint.get("CreationTime"),
            "endpoint_config_name": endpoint["EndpointConfigName"],
            "variants": [dict(
                variant_name=v["VariantName"], model_name=v["ModelName"],
                instance_type=v.get("InstanceType"),
                instance_count=live.get(v["VariantName"], {}).get("CurrentInstanceCount"),
            ) for v in config["ProductionVariants"]],
        }

    def list_endpoints(self):
        return [ep["EndpointName"]
                for page in self.sm_client.get_paginator("list_endpoints").paginate()
                for ep in page.get("Endpoints", [])]

    def delete_endpoint(self, endpoint_name):
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
        self.sm_client.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
        logger.info("Endpoint deleted: %s", endpoint_name)


def main():
    load_dotenv()
    job_name = os.getenv("TRAINING_JOB_NAME")
    if not job_name:
        raise ValueError("Set TRAINING_JOB_NAME to the completed training job.")
    deployer = SageMakerDeployment()
    job = deployer.sm_client.describe_training_job(TrainingJobName=job_name)
    if job["TrainingJobStatus"] != "Completed":
        raise ValueError(f"Training job is not completed: {job['TrainingJobStatus']}")
    model_uri = job["ModelArtifacts"]["S3ModelArtifacts"]
    print("Training job:", job_name)
    print("Deploying artifact:", model_uri)
    predictor = deployer.deploy_model(
        model_uri, endpoint_name=os.getenv("ENDPOINT_NAME") or None
    )
    print("Model deployed. Endpoint:", predictor.endpoint_name)


if __name__ == "__main__":
    main()
