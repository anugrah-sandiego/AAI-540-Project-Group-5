"""
End-to-end test of the training pipeline on a small synthetic dataset.

Runs in a temporary directory with its own MLflow database, model folder and
splits, so it never touches the project's mlflow.db or models/.
"""

import json
import os
import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines import training_pipeline
from src.utils.lineage import file_sha256, UncommittedCodeError, ALLOW_DIRTY_ENV_VAR
from tests.test_data_preprocessing import make_raw_df


def write_inputs(tmp_path, allow_dirty):
    """Write a small synthetic dataset and a minimal config (one model, one strategy, 2 folds)."""
    data_path = tmp_path / 'bank-sample.csv'
    make_raw_df(n=400, seed=1).to_csv(data_path, sep=';', index=False)
    config = {
        'data': {'separator': ';', 'processed_path': 'processed', 'target_column': 'y',
                 'val_size': 0.15, 'test_size': 0.15, 'random_state': 42,
                 'excluded_features': ['duration']},
        'feature_engineering': {'add_domain_features': True},
        'training': {'cv_folds': 2, 'selection_metric': 'average_precision',
                     'candidate_models': ['logistic_regression'], 'imbalance_strategies': ['none'],
                     'tuning_iterations': 1, 'threshold_metric': 'f1',
                     'quality_gates': {}},  # random synthetic data: no gates
        'evaluation': {'lift_percentiles': [10]},
        'model': {'output_dir': 'models'},
        'mlflow': {'experiment_name': 'pipeline_test',
                   'tracking_uri': f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"},
        'governance': {'allow_dirty': allow_dirty},
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return config_path, data_path, config


@pytest.fixture(scope='module')
def pipeline_run(tmp_path_factory):
    """Run the minimal pipeline once and share its outputs."""
    tmp_path = tmp_path_factory.mktemp('pipeline')
    previous_cwd = os.getcwd()
    previous_allow_dirty = os.environ.pop(ALLOW_DIRTY_ENV_VAR, None)
    # Use the config's temporary MLflow store, never one set for the dev container
    previous_tracking_uri = os.environ.pop('MLFLOW_TRACKING_URI', None)
    os.chdir(tmp_path)  # relative outputs (models/, processed/, mlruns/) land here
    # The test checkout may have uncommitted changes (or no git at all, e.g. in Docker),
    # so this run is an explicit experiment
    config_path, data_path, config = write_inputs(tmp_path, allow_dirty=True)

    try:
        result = TrainingPipeline(str(config_path)).run(str(data_path))
        metadata = json.loads((tmp_path / 'models' / 'model_metadata.json').read_text())
        yield {'result': result, 'metadata': metadata, 'data_path': data_path, 'dir': tmp_path,
               'client': MlflowClient(config['mlflow']['tracking_uri'])}
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        os.chdir(previous_cwd)
        if previous_allow_dirty is not None:
            os.environ[ALLOW_DIRTY_ENV_VAR] = previous_allow_dirty
        if previous_tracking_uri is not None:
            os.environ['MLFLOW_TRACKING_URI'] = previous_tracking_uri


def test_pipeline_saves_model_and_metrics(pipeline_run):
    tmp_path = pipeline_run['dir']
    assert (tmp_path / 'models' / 'model.joblib').exists()
    assert {'train', 'val', 'test'} == {p.stem for p in (tmp_path / 'processed').glob('*.parquet')}
    assert 0 <= pipeline_run['metadata']['test_metrics']['roc_auc'] <= 1


def test_metadata_contains_lineage(pipeline_run):
    metadata, data_path = pipeline_run['metadata'], pipeline_run['data_path']
    lineage = metadata['lineage']

    assert metadata['trained_at'] == lineage['collected_at']
    assert lineage['data']['sha256'] == file_sha256(data_path)
    assert lineage['data']['n_rows'] == 400 and lineage['data']['n_columns'] == 17
    assert lineage['git']['source'] in ('git', 'env', 'unavailable')
    assert 'scikit-learn' in lineage['environment']['packages']


def test_metadata_input_schema_matches_model_inputs(pipeline_run):
    schema = pipeline_run['metadata']['input_schema']
    names = [c['name'] for c in schema]
    assert 'duration' not in names and 'y' not in names  # excluded / target columns
    assert len(names) == 15
    assert {c['name']: c['kind'] for c in schema}['job'] == 'categorical'
    assert {c['name']: c['kind'] for c in schema}['age'] == 'numeric'


def test_mlflow_run_has_lineage_tags_artifacts_and_dataset(pipeline_run):
    client, metadata = pipeline_run['client'], pipeline_run['metadata']
    run = client.get_run(pipeline_run['result']['run_id'])
    lineage = metadata['lineage']

    assert run.data.tags['data_sha256'] == lineage['data']['sha256']
    assert run.data.tags['git_commit'] == (lineage['git']['commit'] or 'unknown')
    assert run.data.tags['git_dirty'] == str(lineage['git']['dirty'])
    assert run.data.tags['code_state'] == lineage['code_state'] in ('clean', 'dirty', 'unknown')
    assert run.data.tags['allow_dirty'] == 'True'

    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {'lineage.json', 'input_schema.json'} <= artifacts

    inputs = run.inputs.dataset_inputs
    assert len(inputs) == 1
    assert inputs[0].dataset.name == 'bank-sample'
    assert {t.key: t.value for t in inputs[0].tags}['sha256'] == lineage['data']['sha256']


def fake_dirty_lineage(data_path, repo_dir=None):
    """Lineage for a checkout with uncommitted changes, independent of the real repo state."""
    return {'collected_at': '2026-10-13T00:00:00+00:00',
            'git': {'commit': 'abc123', 'branch': 'aishwarya', 'dirty': True,
                    'dirty_files': ['src/models/train_model.py'], 'source': 'git'},
            'data': {'path': 'bank-sample.csv', 'sha256': 'x', 'size_bytes': 1},
            'environment': {}}


def test_dirty_code_is_refused_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ALLOW_DIRTY_ENV_VAR, raising=False)
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    monkeypatch.setattr(training_pipeline, 'collect_lineage', fake_dirty_lineage)
    config_path, data_path, _ = write_inputs(tmp_path, allow_dirty=False)

    with pytest.raises(UncommittedCodeError, match='src/models/train_model.py'):
        TrainingPipeline(str(config_path)).run(str(data_path))

    # Nothing was trained, saved or logged
    assert not (tmp_path / 'models').exists()
    assert not (tmp_path / 'processed').exists()
    assert not (tmp_path / 'mlflow.db').exists()


def test_env_vars_override_config(tmp_path, monkeypatch):
    """ALLOW_DIRTY and MLFLOW_TRACKING_URI take precedence over config.yaml."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ALLOW_DIRTY_ENV_VAR, '1')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f"sqlite:///{(tmp_path / 'env.db').as_posix()}")
    monkeypatch.setattr(training_pipeline, 'collect_lineage', fake_dirty_lineage)
    config_path, data_path, _ = write_inputs(tmp_path, allow_dirty=False)

    TrainingPipeline(str(config_path)).run(str(data_path))

    lineage = json.loads((tmp_path / 'models' / 'model_metadata.json').read_text())['lineage']
    assert lineage['code_state'] == 'dirty' and lineage['allow_dirty'] is True
    assert (tmp_path / 'env.db').exists() and not (tmp_path / 'mlflow.db').exists()
