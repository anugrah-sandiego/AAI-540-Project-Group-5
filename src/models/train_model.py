"""
Model training module.

Every model is an imbalanced-learn Pipeline (feature engineering -> preprocessing
-> optional SMOTE -> classifier), so the saved artifact accepts raw columns and
SMOTE is only ever applied to the training folds, never to validation data.
"""

import mlflow
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import loguniform, randint, uniform
import pandas as pd
from typing import Dict, Any, List

from src.data_preprocessing.data_preprocessing import build_preprocessor
from src.feature_engineering.feature_engineering import add_domain_features


IMBALANCE_STRATEGIES = ('none', 'class_weight', 'smote')
CV_METRICS = ['roc_auc', 'average_precision', 'f1', 'precision', 'recall']


def _make_classifier(model_type: str, class_weight, random_state: int):
    """Instantiate a classifier with sensible defaults for this dataset."""
    if model_type == 'logistic_regression':
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=random_state)
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                      class_weight=class_weight, random_state=random_state)
    if model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                      class_weight=class_weight, random_state=random_state)
    if model_type == 'gradient_boosting':
        # Histogram-based gradient boosting supports class_weight natively
        return HistGradientBoostingClassifier(class_weight=class_weight, random_state=random_state)
    raise ValueError(f"Unknown model type: {model_type}")


def build_model(model_type: str, numeric_cols: List[str], categorical_cols: List[str],
                imbalance_strategy: str = 'none', add_features: bool = True,
                random_state: int = 42) -> Pipeline:
    """
    Build an end-to-end training pipeline.

    Args:
        model_type: 'logistic_regression', 'decision_tree', 'random_forest' or 'gradient_boosting'
        numeric_cols: Numerical input columns (after feature engineering)
        categorical_cols: Categorical input columns
        imbalance_strategy: 'none', 'class_weight' (cost-sensitive loss) or 'smote' (oversampling)
        add_features: Whether to add domain features before preprocessing
        random_state: Random seed

    Returns:
        Unfitted imblearn Pipeline
    """
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")

    class_weight = 'balanced' if imbalance_strategy == 'class_weight' else None
    steps = []
    if add_features:
        steps.append(('features', FunctionTransformer(add_domain_features)))
    steps.append(('preprocess', build_preprocessor(numeric_cols, categorical_cols)))
    if imbalance_strategy == 'smote':
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('classifier', _make_classifier(model_type, class_weight, random_state)))
    return Pipeline(steps)


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                   numeric_cols: List[str], categorical_cols: List[str],
                   model_types: List[str], imbalance_strategies: List[str],
                   cv_folds: int = 5, add_features: bool = True,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Cross-validate every model / imbalance-strategy combination and log each to MLflow.

    Folds are shuffled because the source data is ordered by date.

    Args:
        X_train: Training features
        y_train: Training target
        numeric_cols: Numerical input columns
        categorical_cols: Categorical input columns
        model_types: Models to compare
        imbalance_strategies: Imbalance strategies to compare
        cv_folds: Number of CV folds
        add_features: Whether to add domain features
        random_state: Random seed

    Returns:
        DataFrame of mean CV metrics, one row per combination, best first
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows = []

    for model_type in model_types:
        for strategy in imbalance_strategies:
            model = build_model(model_type, numeric_cols, categorical_cols,
                                strategy, add_features, random_state)
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=CV_METRICS, n_jobs=-1)
            row = {'model_type': model_type, 'imbalance_strategy': strategy}
            row.update({f'cv_{m}': scores[f'test_{m}'].mean() for m in CV_METRICS})
            rows.append(row)

            with mlflow.start_run(run_name=f"{model_type}__{strategy}", nested=True):
                mlflow.log_params({'model_type': model_type, 'imbalance_strategy': strategy,
                                   'cv_folds': cv_folds})
                mlflow.log_metrics({k: v for k, v in row.items() if k.startswith('cv_')})

            print(f"  {model_type:<20} {strategy:<13} "
                  f"AUC={row['cv_roc_auc']:.4f}  PR-AUC={row['cv_average_precision']:.4f}  "
                  f"F1={row['cv_f1']:.4f}  recall={row['cv_recall']:.4f}")

    return pd.DataFrame(rows).sort_values('cv_average_precision', ascending=False).reset_index(drop=True)


# Bounded search spaces (design document: avoid excessively large searches)
PARAM_DISTRIBUTIONS = {
    'logistic_regression': {
        'classifier__C': loguniform(1e-3, 1e2),
    },
    'decision_tree': {
        'classifier__max_depth': randint(3, 15),
        'classifier__min_samples_leaf': randint(10, 200),
    },
    'random_forest': {
        'classifier__n_estimators': randint(100, 400),
        'classifier__max_depth': [8, 12, 16, None],
        'classifier__min_samples_leaf': randint(1, 20),
        'classifier__max_features': ['sqrt', 0.3, 0.5],
    },
    'gradient_boosting': {
        'classifier__learning_rate': loguniform(0.02, 0.3),
        'classifier__max_iter': randint(100, 500),
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_leaf': randint(20, 200),
        'classifier__l2_regularization': uniform(0, 1),
    },
}


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                          model_type: str, numeric_cols: List[str], categorical_cols: List[str],
                          imbalance_strategy: str = 'none', n_iter: int = 20,
                          scoring: str = 'average_precision', cv_folds: int = 5,
                          add_features: bool = True, random_state: int = 42) -> Dict[str, Any]:
    """
    Tune hyperparameters with a bounded randomized search.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune
        numeric_cols: Numerical input columns
        categorical_cols: Categorical input columns
        imbalance_strategy: Imbalance strategy to use
        n_iter: Number of parameter settings sampled
        scoring: Metric optimised by the search
        cv_folds: Number of CV folds
        add_features: Whether to add domain features
        random_state: Random seed

    Returns:
        Dictionary containing best model, parameters and CV score
    """
    model = build_model(model_type, numeric_cols, categorical_cols,
                        imbalance_strategy, add_features, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(model, PARAM_DISTRIBUTIONS[model_type], n_iter=n_iter,
                                scoring=scoring, cv=cv, n_jobs=-1, random_state=random_state)
    search.fit(X_train, y_train)

    return {
        'best_model': search.best_estimator_,
        'best_params': {k.replace('classifier__', ''): v for k, v in search.best_params_.items()},
        'best_score': search.best_score_,
    }
