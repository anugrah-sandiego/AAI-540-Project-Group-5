"""
Model prediction and evaluation module.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve)
from typing import Dict, List


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get the predicted probability of subscription for each customer.

    Args:
        model: Trained model object
        X: Feature DataFrame

    Returns:
        1-D array of positive-class probabilities
    """
    return model.predict_proba(X)[:, 1]


def predict_model(model, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Make class predictions using a tuned decision threshold.

    Args:
        model: Trained model object
        X: Feature DataFrame
        threshold: Probability above which a customer is predicted to subscribe

    Returns:
        Array of 0/1 predictions
    """
    return (predict_proba(model, X) >= threshold).astype(int)


def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray, metric: str = 'f1') -> Dict[str, float]:
    """
    Choose the decision threshold that maximises F1 on held-out data.

    Should be run on the validation split, never on the test split.

    Args:
        y_true: True labels
        y_proba: Predicted positive-class probabilities
        metric: Metric to maximise (currently 'f1')

    Returns:
        Dictionary with the chosen threshold and its precision, recall and F1
    """
    if metric != 'f1':
        raise ValueError(f"Unsupported threshold metric: {metric}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # The last precision/recall pair has no corresponding threshold
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1))
    return {
        'threshold': float(thresholds[best]),
        'precision': float(precision[best]),
        'recall': float(recall[best]),
        'f1': float(f1[best]),
    }


def lift_at_k(y_true: pd.Series, y_proba: np.ndarray, percentiles: List[int]) -> Dict[str, float]:
    """
    Conversion rate and lift among the top-k% highest-scored customers.

    Lift = conversion rate in the top k% / overall conversion rate. A lift of 3
    means the targeted segment converts three times as often as random outreach.

    Args:
        y_true: True labels
        y_proba: Predicted positive-class probabilities
        percentiles: Top-k percentages to evaluate, e.g. [10, 20, 30]

    Returns:
        Dictionary with conversion_at_k and lift_at_k for each k
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_proba)
    base_rate = y_true.mean()
    results = {}
    for k in percentiles:
        n_top = max(1, int(np.ceil(len(y_true) * k / 100)))
        conversion = y_true[order[:n_top]].mean()
        results[f'conversion_at_{k}'] = float(conversion)
        results[f'lift_at_{k}'] = float(conversion / base_rate)
    return results


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   threshold: float = 0.5, lift_percentiles: List[int] = (10, 20, 30)) -> Dict[str, float]:
    """
    Evaluate model performance with technical and business metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target
        threshold: Decision threshold for class predictions
        lift_percentiles: Top-k percentages for lift / conversion

    Returns:
        Dictionary containing evaluation metrics
    """
    y_proba = predict_proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'true_negatives': int(tn), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_positives': int(tp),
    }
    metrics.update(lift_at_k(y_test, y_proba, list(lift_percentiles)))
    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model object (a bare estimator or a Pipeline ending in 'classifier')
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'named_steps'):
        model = model.named_steps['classifier']

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attribute")

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
