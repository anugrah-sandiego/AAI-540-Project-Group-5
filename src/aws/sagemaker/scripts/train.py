import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import mlflow
import mlflow.sklearn


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
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))

    return parser.parse_args()


def load_data(train_path, val_path):
    """Load training and validation data."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Separate features and target
    target_column = 'y'
    excluded_features = ['duration']

    X_train = train_df.drop(columns=[target_column] + excluded_features)
    y_train = train_df[target_column].map({'yes': 1, 'no': 0})

    X_val = val_df.drop(columns=[target_column] + excluded_features)
    y_val = val_df[target_column].map({'yes': 1, 'no': 0})

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
