"""
Pipeline step: decide whether the challenger may be registered.

Runs as a SageMaker Processing job. The challenger and the current champion
(the latest approved model) are scored on the evaluation batches, which neither
has trained on. The challenger passes if it meets the quality gates from
config.yaml, does not fall below the champion's PR-AUC, and finds subscribers in
every age band at a similar rate (equal opportunity). The decision is written to
evaluation.json, which the pipeline's condition step reads and the registry stores.
"""

import argparse
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.aws.registry import latest_packages, model_data_url
from src.data_preprocessing.data_preprocessing import encode_target
from src.models.predict_model import lift_at_k, predict_proba
from src.utils.helpers import load_config
from src.utils.lineage import PROJECT_ROOT
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

BASE = Path('/opt/ml/processing')
# The challenger may trail the champion's PR-AUC by at most this much (evaluation noise)
PR_AUC_TOLERANCE = 0.01
# Recall in any age band must reach this share of overall recall
MIN_RECALL_RATIO = 0.8
# Age bands with fewer subscribers than this are too small to judge
MIN_BAND_POSITIVES = 10
AGE_BANDS = {'under 30': (0, 30), '30 to 59': (30, 60), '60 and over': (60, 200)}


def load_model(model_tar: Path) -> Tuple[object, dict]:
    """Extract a SageMaker model.tar.gz and load the pipeline and its metadata."""
    target = Path(tempfile.mkdtemp())
    with tarfile.open(model_tar) as archive:
        archive.extractall(target, filter='data')
    return joblib.load(target / 'model.joblib'), json.loads((target / 'model_metadata.json').read_text())


def fetch_champion(sagemaker, s3, group: str) -> Optional[Path]:
    """Download the latest approved model, or None before any model is approved."""
    packages = latest_packages(sagemaker, group, 'Approved')
    if not packages:
        return None
    bucket, key = model_data_url(sagemaker, packages[0]).removeprefix('s3://').split('/', 1)
    target = Path(tempfile.mkdtemp()) / 'model.tar.gz'
    s3.download_file(bucket, key, str(target))
    return target


def score(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    proba = predict_proba(model, X)
    return {'roc_auc': float(roc_auc_score(y, proba)),
            'pr_auc': float(average_precision_score(y, proba)),
            'lift_at_10': lift_at_k(y, proba, [10])['lift_at_10']}


def recall_by_age_band(model, threshold: float, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Share of actual subscribers the model would call, overall and per age band."""
    called = predict_proba(model, X) >= threshold
    recalls = {'overall': float(called[y == 1].mean())}
    for band, (low, high) in AGE_BANDS.items():
        positives = (y == 1) & X['age'].between(low, high, inclusive='left')
        if positives.sum() >= MIN_BAND_POSITIVES:
            recalls[band] = float(called[positives].mean())
    return recalls


def decide(challenger: Dict[str, float], champion: Optional[Dict[str, float]],
           recalls: Dict[str, float], gates: Dict[str, float]) -> List[str]:
    """Reasons to reject the challenger; an empty list means it passes."""
    reasons = [f"{metric} {challenger[metric]:.3f} is below the gate {minimum}"
               for metric, minimum in gates.items() if challenger[metric] < minimum]
    if champion and challenger['pr_auc'] < champion['pr_auc'] - PR_AUC_TOLERANCE:
        reasons.append(f"pr_auc {challenger['pr_auc']:.3f} is below the champion's {champion['pr_auc']:.3f}")
    reasons += [f"recall for age {band} is {recall:.2f}, under {MIN_RECALL_RATIO:.0%} of overall"
                for band, recall in recalls.items()
                if band != 'overall' and recall < MIN_RECALL_RATIO * recalls['overall']]
    return reasons


def evaluate(challenger_tar: Path, champion_tar: Optional[Path], data: pd.DataFrame, config: dict) -> dict:
    """Score both models on the evaluation data and record the decision."""
    target = config['data']['target_column']
    X = data.drop(columns=[target] + config['data']['excluded_features'])
    y = encode_target(data[target])
    if y.nunique() < 2:
        return {'passed': False, 'reasons': ['evaluation data contains only one class']}

    challenger_model, metadata = load_model(challenger_tar)
    challenger = score(challenger_model, X, y)
    champion = score(load_model(champion_tar)[0], X, y) if champion_tar else None
    recalls = recall_by_age_band(challenger_model, metadata['decision_threshold'], X, y)
    reasons = decide(challenger, champion, recalls, config['training']['quality_gates'])
    return {'passed': not reasons, 'reasons': reasons, 'eval_rows': len(data),
            'challenger': challenger, 'champion': champion, 'recall_by_age_band': recalls}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-group', required=True)
    parser.add_argument('--region', required=True)
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    champion_tar = fetch_champion(session.client('sagemaker'), session.client('s3'), args.model_group)
    result = evaluate(BASE / 'challenger' / 'model.tar.gz', champion_tar,
                      pd.read_parquet(BASE / 'eval' / 'eval.parquet'), load_config(PROJECT_ROOT / 'config.yaml'))
    logger.info("Challenger %s: %s", 'passed' if result['passed'] else 'rejected', result)

    (BASE / 'evaluation').mkdir(parents=True, exist_ok=True)
    (BASE / 'evaluation' / 'evaluation.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
