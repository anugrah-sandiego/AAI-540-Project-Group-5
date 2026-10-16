"""Tests for the SageMaker pipeline steps (prepare, train, evaluate) without AWS."""

import tarfile
from pathlib import Path

import pandas as pd
import pytest
from botocore.stub import Stubber
import boto3

from src.aws.steps.evaluate import decide, evaluate, fetch_champion, recall_by_age_band, load_model
from src.aws.steps.prepare import prepare
from src.aws.steps.train import QUICK_TRAINING, training_config
from src.utils.helpers import load_config
from tests.model_fixtures import save_model_dir
from tests.test_data_preprocessing import make_raw_df


@pytest.fixture
def model_dir(tmp_path):
    return save_model_dir(tmp_path / 'model')


def write_batch(raw_dir: Path, batch_id: str, frame: pd.DataFrame) -> None:
    (raw_dir / batch_id).mkdir(parents=True)
    frame.to_csv(raw_dir / batch_id / 'part.csv', sep=';', index=False)


# --- prepare ------------------------------------------------------------------

def test_first_run_holds_out_the_newest_batch(tmp_path):
    write_batch(tmp_path, 'batch_id=b0000', make_raw_df(n=100, seed=1))
    write_batch(tmp_path, 'batch_id=b0001', make_raw_df(n=40, seed=2))
    train, evaluation, report, status = prepare(tmp_path, checkpoint={})
    assert (len(train), len(evaluation)) == (100, 40)
    assert report['eval_batches'] == ['batch_id=b0001']
    assert {batch: entry['status'] for batch, entry in status.items()} == {
        'batch_id=b0000': 'processed', 'batch_id=b0001': 'processed'}


def test_later_runs_evaluate_only_the_new_batches(tmp_path):
    write_batch(tmp_path, 'batch_id=b0000', make_raw_df(n=100, seed=1))
    _, _, _, checkpoint = prepare(tmp_path, checkpoint={})
    write_batch(tmp_path, 'batch_id=b0001', make_raw_df(n=40, seed=2))
    train, evaluation, report, _ = prepare(tmp_path, checkpoint)
    assert (len(train), len(evaluation)) == (100, 40)
    assert report['new_batches'] == ['batch_id=b0001']


def test_invalid_batch_is_quarantined_once_and_skipped_after(tmp_path):
    write_batch(tmp_path, 'batch_id=b0000', make_raw_df(n=100, seed=1))
    write_batch(tmp_path, 'batch_id=b0001', make_raw_df(n=40, seed=2).drop(columns='age'))
    _, _, report, checkpoint = prepare(tmp_path, checkpoint={})
    assert 'age' in report['quarantined']['batch_id=b0001']
    assert checkpoint['batch_id=b0001']['status'] == 'quarantined'
    _, _, report, _ = prepare(tmp_path, checkpoint)
    assert report['quarantined'] == {}


def test_duplicate_rows_across_batches_are_dropped(tmp_path):
    batch = make_raw_df(n=60, seed=1)
    write_batch(tmp_path, 'batch_id=b0000', batch)
    write_batch(tmp_path, 'batch_id=b0001', pd.concat([batch.head(10), make_raw_df(n=30, seed=2)]))
    train, evaluation, report, _ = prepare(tmp_path, checkpoint={})
    assert report['duplicates_dropped'] == 10
    assert len(train) + len(evaluation) == 90


def test_no_valid_batches_fails(tmp_path):
    write_batch(tmp_path, 'batch_id=b0000', make_raw_df(n=20).drop(columns='y'))
    with pytest.raises(ValueError, match='No valid batches'):
        prepare(tmp_path, checkpoint={})


# --- train --------------------------------------------------------------------

def test_training_config_targets_sagemaker_directories(tmp_path):
    config = training_config(load_config('config.yaml'), quick=True, ml_root=tmp_path)
    assert config['model']['output_dir'] == str(tmp_path / 'model')
    assert config['mlflow']['tracking_uri'] == f"sqlite:///{tmp_path / 'output' / 'data' / 'mlflow.db'}"
    assert {key: config['training'][key] for key in QUICK_TRAINING} == QUICK_TRAINING
    base = load_config('config.yaml')
    full = training_config(base, quick=False, ml_root=tmp_path)
    assert full['training']['candidate_models'] == base['training']['candidate_models']


def test_training_leaves_the_quality_gates_to_the_evaluate_step(tmp_path):
    base = load_config('config.yaml')
    assert training_config(base, quick=False, ml_root=tmp_path)['training']['quality_gates'] == {}
    # The evaluate step still reads the gates from the unchanged config
    assert base['training']['quality_gates']


# --- evaluate -----------------------------------------------------------------

GATES = {'roc_auc': 0.75, 'lift_at_10': 2.5}
GOOD = {'roc_auc': 0.80, 'pr_auc': 0.45, 'lift_at_10': 4.0}


def test_decide_passes_a_challenger_that_meets_every_gate():
    assert decide(GOOD, {'pr_auc': 0.455}, {'overall': 0.5, 'under 30': 0.45}, GATES) == []


@pytest.mark.parametrize('challenger, champion, recalls, reason', [
    (GOOD | {'roc_auc': 0.70}, None, {'overall': 0.5}, 'roc_auc 0.700 is below the gate 0.75'),
    (GOOD, {'pr_auc': 0.50}, {'overall': 0.5}, "below the champion's 0.500"),
    (GOOD, None, {'overall': 0.5, '60 and over': 0.3}, 'recall for age 60 and over is 0.30'),
])
def test_decide_explains_each_rejection(challenger, champion, recalls, reason):
    reasons = decide(challenger, champion, recalls, GATES)
    assert len(reasons) == 1 and reason in reasons[0]


def test_recall_by_age_band_skips_small_bands(model_dir):
    model, metadata = load_model(tar_model(model_dir))
    data = make_raw_df(n=300, seed=3)
    X, y = data.drop(columns=['y', 'duration']), (data['y'] == 'yes').astype(int)
    recalls = recall_by_age_band(model, metadata['decision_threshold'], X, y)
    assert 'overall' in recalls
    assert all(0 <= value <= 1 for value in recalls.values())


def tar_model(model_dir: Path) -> Path:
    archive_path = model_dir.parent / f'{model_dir.name}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in ('model.joblib', 'model_metadata.json'):
            archive.add(model_dir / name, arcname=name)
    return archive_path


def test_evaluate_compares_challenger_with_champion(tmp_path):
    challenger = tar_model(save_model_dir(tmp_path / 'challenger', seed=0))
    champion = tar_model(save_model_dir(tmp_path / 'champion', seed=1))
    result = evaluate(challenger, champion, make_raw_df(n=200, seed=4), load_config('config.yaml'))
    assert set(result['challenger']) == set(result['champion']) == {'roc_auc', 'pr_auc', 'lift_at_10'}
    assert result['passed'] == (result['reasons'] == [])
    assert result['eval_rows'] == 200


def test_evaluate_rejects_single_class_data(model_dir):
    data = make_raw_df(n=50).assign(y='no')
    result = evaluate(tar_model(model_dir), None, data, load_config('config.yaml'))
    assert result == {'passed': False, 'reasons': ['evaluation data contains only one class']}


def test_no_champion_before_any_approval():
    sagemaker = boto3.client('sagemaker', region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x')
    with Stubber(sagemaker) as stub:
        stub.add_response('list_model_packages', {'ModelPackageSummaryList': []})
        assert fetch_champion(sagemaker, s3=None, group='bank-marketing-models') is None
