"""
main's cloud training: one random forest in AWS's scikit-learn container.

Uploads the local processed splits to S3 (main expected them there but nothing
uploaded them), then trains with Anugrah Rastogi's script (src/aws/sklearn/train.py)
and hyperparameters from main's src/aws/sagemaker/training.py. The model is saved
to S3 but not registered: it is a different model format from the pipeline's
image, which serves the registry's champion.
"""

from pathlib import Path

from sagemaker.inputs import TrainingInput
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn

from src.aws.config import AwsConfig

SPLITS = Path('data/processed')
HYPERPARAMETERS = {'n_estimators': 314, 'max_depth': 16, 'min_samples_leaf': 8, 'max_features': 0.5,
                   'random_state': 42}


def train_sklearn(cfg: AwsConfig, aws, instance_type: str) -> str:
    """Train and return the S3 URI of model.tar.gz; the job log prints its validation metrics."""
    s3 = aws.client('s3')
    channels = {}
    for channel, name in (('train', 'train'), ('validation', 'val')):
        s3.upload_file(str(SPLITS / f'{name}.parquet'), cfg.bucket, cfg.key('processed', f'{name}.parquet'))
        channels[channel] = TrainingInput(cfg.s3('processed', f'{name}.parquet'),
                                          content_type='application/x-parquet')
    estimator = SKLearn(entry_point='train.py', source_dir=str(Path(__file__).parent / 'sklearn'),
                        role=cfg.role_arn, instance_type=instance_type, instance_count=1,
                        framework_version='1.2-1', py_version='py3', hyperparameters=HYPERPARAMETERS,
                        output_path=cfg.s3('sklearn-models'), base_job_name='bank-marketing-sklearn',
                        sagemaker_session=Session(boto_session=aws))
    estimator.fit(channels, wait=True, logs=True)
    return estimator.model_data
