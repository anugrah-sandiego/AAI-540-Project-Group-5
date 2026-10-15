"""
Pipeline step: train a challenger with the project's training pipeline.

SageMaker starts a custom training image as `train` (the `docker/train` script),
which runs this module. It reads the prepared training data from the `train`
channel and writes model.joblib and model_metadata.json to /opt/ml/model, which
SageMaker packages as model.tar.gz. MLflow runs and splits go to the output data
directory, which SageMaker uploads to S3 alongside the model.
"""

import copy
import json
import os
import tempfile
from pathlib import Path

import yaml

from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.helpers import load_config
from src.utils.lineage import PROJECT_ROOT
from src.utils.logging_config import setup_logging

ML_ROOT = Path('/opt/ml')

# Runs compare all models by default; setting the `quick` hyperparameter to true
# reuses the known-good model family with a short search (cheaper test runs)
QUICK_TRAINING = {'candidate_models': ['random_forest'], 'imbalance_strategies': ['none'],
                  'tuning_iterations': 5, 'cv_folds': 3}


def training_config(base: dict, quick: bool, ml_root: Path = ML_ROOT) -> dict:
    """Point the pipeline's outputs at SageMaker's directories."""
    config = copy.deepcopy(base)
    output = ml_root / 'output' / 'data'
    config['model']['output_dir'] = str(ml_root / 'model')
    config['data']['processed_path'] = str(output / 'processed')
    config['mlflow']['tracking_uri'] = f"sqlite:///{output / 'mlflow.db'}"
    if quick:
        config['training'].update(QUICK_TRAINING)
    return config


def main() -> None:
    setup_logging()
    hyperparameters_file = ML_ROOT / 'input' / 'config' / 'hyperparameters.json'
    hyperparameters = json.loads(hyperparameters_file.read_text()) if hyperparameters_file.exists() else {}
    quick = str(hyperparameters.get('quick', 'false')).lower() == 'true'

    config = training_config(load_config(PROJECT_ROOT / 'config.yaml'), quick)
    config_path = Path(tempfile.gettempdir()) / 'sagemaker-config.yaml'
    config_path.write_text(yaml.safe_dump(config))

    # MLflow stores run artifacts under ./mlruns, so run from the uploaded output directory
    output_dir = ML_ROOT / 'output' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = ML_ROOT / 'input' / 'data' / 'train' / 'train.parquet'
    os.chdir(output_dir)
    TrainingPipeline(str(config_path)).run(str(data_path))


if __name__ == '__main__':
    main()
