"""
Training pipeline for end-to-end model training.

Steps: load -> validate -> stratified 70/15/15 split -> compare models and
class-imbalance strategies with CV -> tune the best candidate -> choose the
decision threshold on validation -> evaluate once on test -> quality gates ->
log, register and save the model.
"""

import json
import joblib
import mlflow
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

        # Load and validate data
        print("Loading data...")
        df = load_data(data_path, sep=data_cfg.get('separator', ','))
        quality_report = validate_schema(df)
        print(f"Data quality: {quality_report}")

        # Drop leakage columns and split
        excluded = data_cfg.get('excluded_features', [])
        X = df.drop(columns=[target_column] + excluded)
        y = encode_target(df[target_column])
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, data_cfg['val_size'], data_cfg['test_size'], random_state
        )
        print(f"Split sizes - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
        self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

        numeric_cols = [c for c in NUMERIC_COLUMNS if c not in excluded]
        if add_features:
            numeric_cols += DOMAIN_FEATURES
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c not in excluded]

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name='training_pipeline') as run:
            mlflow.log_params({'excluded_features': ','.join(excluded),
                               'add_domain_features': add_features,
                               'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test)})
            mlflow.log_dict(quality_report, 'data_quality_report.json')

            # Compare models and class-imbalance strategies
            print("Comparing models and imbalance strategies (CV on train)...")
            comparison = compare_models(
                X_train, y_train, numeric_cols, categorical_cols,
                train_cfg['candidate_models'], train_cfg['imbalance_strategies'],
                train_cfg['cv_folds'], add_features, random_state
            )
            mlflow.log_table(comparison, 'model_comparison.json')
            best = comparison.iloc[0]
            print(f"Best candidate: {best.model_type} / {best.imbalance_strategy}")

            # Tune the best candidate
            print("Tuning hyperparameters...")
            tuned = hyperparameter_tuning(
                X_train, y_train, best.model_type, numeric_cols, categorical_cols,
                best.imbalance_strategy, train_cfg['tuning_iterations'],
                train_cfg['selection_metric'], train_cfg['cv_folds'], add_features, random_state
            )
            model = tuned['best_model']
            print(f"Best params: {tuned['best_params']} (CV {train_cfg['selection_metric']}: "
                  f"{tuned['best_score']:.4f})")

            # Choose the decision threshold on validation data
            threshold_info = find_best_threshold(y_val, predict_proba(model, X_val),
                                                 train_cfg['threshold_metric'])
            threshold = threshold_info['threshold']
            print(f"Validation threshold: {threshold:.4f} -> {threshold_info}")

            # Final, single evaluation on the untouched test set
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

            mlflow.sklearn.log_model(
                model, name='model', serialization_format='cloudpickle',
                signature=infer_signature(X_train.head(100), predict_proba(model, X_train.head(100))),
                input_example=X_train.head(3),
            )

            self._print_report(threshold, default_metrics, test_metrics, percentiles)
            if failed_gates:
                raise RuntimeError(f"Model failed quality gates {gates}: {failed_gates}")

            metadata = {
                'run_id': run.info.run_id,
                'model_type': best.model_type,
                'imbalance_strategy': best.imbalance_strategy,
                'params': tuned['best_params'],
                'decision_threshold': threshold,
                'excluded_features': excluded,
                'test_metrics': test_metrics,
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

    def _save_model(self, model, metadata: dict):
        """Save the fitted pipeline and its metadata (threshold, metrics) for serving."""
        out_dir = Path(self.config['model']['output_dir'])
        ensure_dir(out_dir)
        joblib.dump(model, out_dir / 'model.joblib')
        with open(out_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Model saved to {out_dir / 'model.joblib'}")

    @staticmethod
    def _print_report(threshold, default_metrics, test_metrics, percentiles):
        """Print a short test-set summary."""
        print("\n=== Test set results ===")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}   PR-AUC: {test_metrics['pr_auc']:.4f}")
        for label, m in [('threshold 0.50', default_metrics),
                         (f'tuned {threshold:.3f}', test_metrics)]:
            print(f"{label:>15}: precision={m['precision']:.4f} recall={m['recall']:.4f} "
                  f"F1={m['f1']:.4f}")
        for k in percentiles:
            print(f"Top {k}%: conversion={test_metrics[f'conversion_at_{k}']:.1%} "
                  f"lift={test_metrics[f'lift_at_{k}']:.2f}x")


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    data_path = pipeline.config.get('data', {}).get('raw_path', 'data/raw/bank-full.csv')
    pipeline.run(data_path=data_path)
