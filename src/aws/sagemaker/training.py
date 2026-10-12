"""
SageMaker training job implementation for bank marketing model.
"""

import os
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.model_monitor import DataCaptureConfig
from dotenv import load_dotenv
import logging
from src.utils.logging_config import setup_logging
from pathlib import Path

# Logging is configured once by setup_logging() when this file is run directly
logger = logging.getLogger(__name__)


class SageMakerTrainingJob:
    """Manage SageMaker training jobs for the bank marketing model."""

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

        logger.info(f"Initialized SageMaker training for region: {self.region}")

    def prepare_training_script(self):
        """
        Prepare the training script for SageMaker.

        This script will be uploaded to S3 and used by SageMaker for training.
        """
        training_script_content = '''
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=16)
    parser.add_argument('--min_samples_leaf', type=int, default=8)
    parser.add_argument('--max_features', type=float, default=0.5)
    parser.add_argument('--random_state', type=int, default=42)

    # Data paths
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    return parser.parse_args()


def load_data(train_path, val_path):
    """Load training and validation data."""
    from pathlib import Path

    def read_channel(channel):
        files = sorted(Path(channel).glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'No Parquet input files in {channel}')
        return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    def separate(frame):
        labels = frame['y']
        if labels.isin([0, 1]).all():
            labels = labels.astype(int)
        elif labels.isin(['yes', 'no']).all():
            labels = labels.map({'yes': 1, 'no': 0})
        else:
            raise ValueError('Labels must be 0/1 or yes/no')
        features = frame.drop(columns=['y']).drop(columns=['duration'], errors='ignore')
        return features, labels

    X_train, y_train = separate(read_channel(train_path))
    X_val, y_val = separate(read_channel(val_path))

    return X_train, y_train, X_val, y_val


def preprocess_data(X_train, X_val):
    """Preprocess categorical and numerical features."""
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    # One-hot encode categorical features
    X_train_processed = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_val_processed = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)

    # Align columns (ensure validation has same columns as training)
    for col in X_train_processed.columns:
        if col not in X_val_processed.columns:
            X_val_processed[col] = 0

    X_val_processed = X_val_processed[X_train_processed.columns]

    return X_train_processed, X_val_processed


def train_model(X_train, y_train, args):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba),
        'pr_auc': average_precision_score(y_val, y_proba)
    }

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load data
    X_train, y_train, X_val, y_val = load_data(args.train, args.validation)

    # Preprocess
    X_train_processed, X_val_processed = preprocess_data(X_train, X_val)

    # Train model
    model = train_model(X_train_processed, y_train, args)

    # Evaluate
    metrics = evaluate_model(model, X_val_processed, y_val)

    # Log metrics
    print(f"Training metrics: {metrics}")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
'''

        # Save training script
        script_dir = Path("src/aws/sagemaker/scripts")
        script_dir.mkdir(parents=True, exist_ok=True)

        script_path = script_dir / "train.py"
        with open(script_path, 'w') as f:
            f.write(training_script_content)

        # Installed inside the managed training container, not just Studio.
        (script_dir / 'requirements.txt').write_text('pyarrow==14.0.2\n')

        logger.info(f"Training script saved to {script_path}")
        return str(script_path)

    def launch_training_job(self, train_s3_uri: str, val_s3_uri: str,
                           job_name: str = None, hyperparameters: dict = None):
        """
        Launch a SageMaker training job.

        Args:
            train_s3_uri: S3 URI for training data
            val_s3_uri: S3 URI for validation data
            job_name: Name for the training job
            hyperparameters: Dictionary of hyperparameters

        Returns:
            Training estimator object
        """
        if job_name is None:
            from datetime import datetime
            job_name = f"bank-marketing-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 314,
                'max_depth': 16,
                'min_samples_leaf': 8,
                'max_features': 0.5,
                'random_state': 42
            }

        # Prepare training script
        script_path = self.prepare_training_script()

        # Create training inputs
        train_input = TrainingInput(s3_data=train_s3_uri, content_type='application/x-parquet')
        val_input = TrainingInput(s3_data=val_s3_uri, content_type='application/x-parquet')

        # Create SKLearn estimator
        estimator = SKLearn(
            entry_point='train.py',
            source_dir=str(Path(script_path).parent),
            role=self.role_arn,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            framework_version='1.2-1',
            py_version='py3',
            hyperparameters=hyperparameters,
            output_path=f"s3://{self.bucket_name}/{self.base_prefix}/models/",
            base_job_name='bank-marketing',
            sagemaker_session=self.sagemaker_session
        )

        logger.info(f"Launching training job: {job_name}")
        estimator.fit({'train': train_input, 'validation': val_input}, job_name=job_name)

        logger.info(f"Training job {job_name} completed")
        return estimator

    def get_training_job_status(self, job_name: str) -> dict:
        """
        Get the status of a training job.

        Args:
            job_name: Name of the training job

        Returns:
            Dictionary with job status information
        """
        response = self.sm_client.describe_training_job(TrainingJobName=job_name)
        return {
            'status': response['TrainingJobStatus'],
            'start_time': response.get('TrainingStartTime'),
            'end_time': response.get('TrainingEndTime'),
            'training_time': response.get('TrainingTimeInSeconds'),
            'billable_time': response.get('BillableTimeInSeconds')
        }


def main():
    """Main function to launch training job."""
    trainer = SageMakerTrainingJob()

    # Example S3 URIs (these would be obtained from data upload)
    train_s3_uri = f"s3://{trainer.bucket_name}/{trainer.base_prefix}/processed/train.parquet"
    val_s3_uri = f"s3://{trainer.bucket_name}/{trainer.base_prefix}/processed/val.parquet"

    # Launch training job
    estimator = trainer.launch_training_job(train_s3_uri, val_s3_uri)

    # Log the job name so the run can be tracked in SageMaker and CloudWatch
    logger.info("Training job launched: %s", estimator._current_job_name)


if __name__ == "__main__":
    # Set up shared logging only when run as a script, not on import
    setup_logging()
    main()
