"""
Training pipeline for end-to-end model training.

Steps: collect lineage -> load -> validate -> stratified 70/15/15 split ->
compare models and class-imbalance strategies with CV -> tune the best
candidate -> choose the decision threshold on validation -> evaluate once on
test -> quality gates -> log, register and save the model with its lineage.
"""

import json
import logging
import os
import sys
import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
from src.data_ingestion.data_ingestion import (load_data, validate_schema, split_data,
                                               NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
from src.data_preprocessing.data_preprocessing import encode_target
from src.feature_engineering.feature_engineering import DOMAIN_FEATURES
from src.models.train_model import compare_models, hyperparameter_tuning
from src.models.predict_model import predict_proba, find_best_threshold, evaluate_model
from src.utils.helpers import load_config, ensure_dir
from src.utils.logging_config import setup_logging, log_stage
from src.utils.lineage import (collect_lineage, input_schema, resolve_allow_dirty,
                               enforce_clean_code, UncommittedCodeError)

# Module logger; output format is set by setup_logging() in the entry point
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for term deposit prediction."""

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the training pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)

    def run(self, data_path: str, target_column: str = 'y'):
        """
        Run the complete training pipeline.

        Args:
            data_path: Path to raw data file
            target_column: Name of target column

        Returns:
            Dictionary containing trained model, threshold and metrics
        """
        data_cfg = self.config['data']
        train_cfg = self.config['training']
        add_features = self.config.get('feature_engineering', {}).get('add_domain_features', True)
        random_state = data_cfg.get('random_state', 42)

        # Record which code, data and environment this run uses, before any training
        # (each stage below is timed and logged by log_stage)
        with log_stage(logger, 'collect lineage'):
            lineage = collect_lineage(data_path)
            git = lineage['git']
            logger.info("Lineage: commit %s (branch %s, source %s), data sha256 %s",
                        git['commit'], git['branch'], git['source'], lineage['data']['sha256'])
            # Stop here, before any training, if the code cannot be traced to a commit
            allow_dirty = resolve_allow_dirty(
                self.config.get('governance', {}).get('allow_dirty', False))
            lineage['code_state'] = enforce_clean_code(git, allow_dirty)
            lineage['allow_dirty'] = allow_dirty

        # Load and validate data
        with log_stage(logger, 'load and validate data'):
            df = load_data(data_path, sep=data_cfg.get('separator', ','))
            quality_report = validate_schema(df)
            logger.info("Data quality: %s", quality_report)
            lineage['data'].update(n_rows=len(df), n_columns=df.shape[1])

        # Drop leakage columns and split
        with log_stage(logger, 'split data'):
            excluded = data_cfg.get('excluded_features', [])
            X = df.drop(columns=[target_column] + excluded)
            y = encode_target(df[target_column])
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                X, y, data_cfg['val_size'], data_cfg['test_size'], random_state
            )
            logger.info("Split sizes - train: %d, val: %d, test: %d",
                        len(X_train), len(X_val), len(X_test))
            self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
            # Columns and types the saved model expects at prediction time
            schema = input_schema(X_train)

        numeric_cols = [c for c in NUMERIC_COLUMNS if c not in excluded]
        if add_features:
            numeric_cols += DOMAIN_FEATURES
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c not in excluded]

        # The standard MLFLOW_TRACKING_URI variable overrides config.yaml, e.g. so the
        # dev container keeps its own store (see docker-compose.yml)
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI') or self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name='training_pipeline') as run:
            # Log the run id so this run is easy to find in the MLflow UI
            logger.info("MLflow run id: %s", run.info.run_id)
            mlflow.log_params({'excluded_features': ','.join(excluded),
                               'add_domain_features': add_features,
                               'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test)})
            mlflow.log_dict(quality_report, 'data_quality_report.json')
            self._log_lineage(lineage, schema, df, target_column)

            # Compare models and class-imbalance strategies
            with log_stage(logger, 'compare models and imbalance strategies (CV on train)'):
                comparison = compare_models(
                    X_train, y_train, numeric_cols, categorical_cols,
                    train_cfg['candidate_models'], train_cfg['imbalance_strategies'],
                    train_cfg['cv_folds'], add_features, random_state
                )
                mlflow.log_table(comparison, 'model_comparison.json')
                best = comparison.iloc[0]
                logger.info("Best candidate: %s / %s", best.model_type, best.imbalance_strategy)

            # Tune the best candidate
            with log_stage(logger, f'tune hyperparameters ({best.model_type})'):
                tuned = hyperparameter_tuning(
                    X_train, y_train, best.model_type, numeric_cols, categorical_cols,
                    best.imbalance_strategy, train_cfg['tuning_iterations'],
                    train_cfg['selection_metric'], train_cfg['cv_folds'], add_features, random_state
                )
                model = tuned['best_model']
                logger.info("Best params: %s (CV %s: %.4f)", tuned['best_params'],
                            train_cfg['selection_metric'], tuned['best_score'])

            # Choose the decision threshold on validation data
            with log_stage(logger, 'choose decision threshold (validation)'):
                threshold_info = find_best_threshold(y_val, predict_proba(model, X_val),
                                                     train_cfg['threshold_metric'])
                threshold = threshold_info['threshold']
                logger.info("Validation threshold: %.4f -> %s", threshold, threshold_info)

            # Final, single evaluation on the untouched test set
            with log_stage(logger, 'evaluate on test set'):
                percentiles = self.config['evaluation']['lift_percentiles']
                val_metrics = evaluate_model(model, X_val, y_val, threshold, percentiles)
                test_metrics = evaluate_model(model, X_test, y_test, threshold, percentiles)
                default_metrics = evaluate_model(model, X_test, y_test, 0.5, percentiles)

                mlflow.log_params({'model_type': best.model_type,
                                   'imbalance_strategy': best.imbalance_strategy,
                                   'decision_threshold': threshold, **tuned['best_params']})
                mlflow.log_metric(f"cv_{train_cfg['selection_metric']}", tuned['best_score'])
                mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
                mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
                mlflow.log_metrics({f'test_default_threshold_{k}': default_metrics[k]
                                    for k in ('precision', 'recall', 'f1')})

            # Quality gates
            gates = train_cfg.get('quality_gates', {})
            failed_gates = {k: test_metrics[k] for k, v in gates.items() if test_metrics[k] < v}
            mlflow.set_tag('quality_gates_passed', str(not failed_gates))

            with log_stage(logger, 'log model to MLflow'):
                mlflow.sklearn.log_model(
                    model, name='model', serialization_format='cloudpickle',
                    signature=infer_signature(X_train.head(100), predict_proba(model, X_train.head(100))),
                    input_example=X_train.head(3),
                )

            self._log_report(threshold, default_metrics, test_metrics, percentiles)
            # Log the gate result; failures are logged before raising so they show up in the logs
            if failed_gates:
                logger.error("Model failed quality gates %s: %s", gates, failed_gates)
                raise RuntimeError(f"Model failed quality gates {gates}: {failed_gates}")
            logger.info("Quality gates passed: %s", gates)

            metadata = {
                'run_id': run.info.run_id,
                'model_type': best.model_type,
                'imbalance_strategy': best.imbalance_strategy,
                'params': tuned['best_params'],
                'decision_threshold': threshold,
                'excluded_features': excluded,
                'test_metrics': test_metrics,
                # Trace the saved model back to its code, data and environment
                'trained_at': lineage['collected_at'],
                'lineage': lineage,
                'input_schema': schema,
            }
            self._save_model(model, metadata)

        return {'model': model, 'threshold': threshold, 'comparison': comparison,
                'test_metrics': test_metrics, 'run_id': run.info.run_id}

    def _save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Persist the splits as Parquet so notebooks and later stages reuse identical data."""
        out_dir = Path(self.config['data']['processed_path'])
        ensure_dir(out_dir)
        target = self.config['data'].get('target_column', 'y')
        for name, X_part, y_part in [('train', X_train, y_train), ('val', X_val, y_val),
                                     ('test', X_test, y_test)]:
            X_part.assign(**{target: y_part}).to_parquet(out_dir / f'{name}.parquet', index=False)
        # Confirm where the splits were written
        logger.info("Splits saved to %s", out_dir)

    @staticmethod
    def _log_lineage(lineage: dict, schema: list, df, target_column: str):
        """Record lineage in the active MLflow run: searchable tags, JSON artifacts and the dataset."""
        git, data = lineage['git'], lineage['data']
        # Tags are searchable in the MLflow UI, e.g. tags.data_sha256 = '...'
        mlflow.set_tags({
            'git_commit': git['commit'] or 'unknown',
            'git_branch': git['branch'] or 'unknown',
            'git_dirty': str(git['dirty']),
            'git_source': git['source'],
            'data_sha256': data['sha256'],
            # Lets a registry accept only models trained from committed code
            'code_state': lineage['code_state'],
            'allow_dirty': str(lineage['allow_dirty']),
        })
        mlflow.log_dict(lineage, 'lineage.json')
        mlflow.log_dict(schema, 'input_schema.json')
        # Register the raw dataset with the run so it shows under "Datasets" in the UI
        dataset = mlflow.data.from_pandas(df, source=data['path'], name=Path(data['path']).stem,
                                          targets=target_column)
        mlflow.log_input(dataset, context='training', tags={'sha256': data['sha256']})

    def _save_model(self, model, metadata: dict):
        """Save the fitted pipeline and its metadata (threshold, metrics) for serving."""
        out_dir = Path(self.config['model']['output_dir'])
        ensure_dir(out_dir)
        joblib.dump(model, out_dir / 'model.joblib')
        with open(out_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        # Confirm where the model was written
        logger.info("Model saved to %s", out_dir / 'model.joblib')

    @staticmethod
    def _log_report(threshold, default_metrics, test_metrics, percentiles):
        """Log a short test-set summary."""
        logger.info("=== Test set results ===")
        logger.info("ROC-AUC: %.4f   PR-AUC: %.4f", test_metrics['roc_auc'], test_metrics['pr_auc'])
        for label, m in [('threshold 0.50', default_metrics),
                         (f'tuned {threshold:.3f}', test_metrics)]:
            logger.info("%15s: precision=%.4f recall=%.4f F1=%.4f",
                        label, m['precision'], m['recall'], m['f1'])
        for k in percentiles:
            logger.info("Top %s%%: conversion=%.1f%% lift=%.2fx",
                        k, 100 * test_metrics[f'conversion_at_{k}'], test_metrics[f'lift_at_{k}'])


if __name__ == '__main__':
    # Configure logging once; on failure log the traceback a single time and exit non-zero
    setup_logging()
    try:
        pipeline = TrainingPipeline()
        data_path = pipeline.config.get('data', {}).get('raw_path', 'data/raw/bank-full.csv')
        pipeline.run(data_path=data_path)
    except UncommittedCodeError as exc:
        # An expected policy refusal: a clear message is enough, no traceback
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Training pipeline failed")
        sys.exit(1)
