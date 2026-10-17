"""Train once on train; tune threshold on validation; evaluate once on test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def separate(frame):
    if 'y' not in frame or frame.empty:
        raise ValueError('Nonempty data with y is required')
    y = frame['y']
    if y.isin(['yes', 'no']).all():
        y = y.map({'no': 0, 'yes': 1})
    elif not y.isin([0, 1]).all():
        raise ValueError('Target must be yes/no or 0/1')
    if y.nunique() != 2:
        raise ValueError('Each split must contain both target classes')
    return frame.drop(columns=['y', 'duration'], errors='ignore'), y.astype(int)


def build_pipeline(frame, trees=314):
    numeric = frame.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in frame if c not in numeric]
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])
    model = Pipeline([('preprocess', pre), ('classifier', RandomForestClassifier(
        n_estimators=trees, max_depth=16, min_samples_leaf=8, max_features=0.5,
        class_weight='balanced', random_state=42, n_jobs=-1))])
    return model, numeric, categorical


def metrics(y, p, threshold):
    n = max(1, int(np.ceil(len(y) * .1)))
    order = np.argsort(-p, kind='stable')[:n]
    return {'roc_auc': float(roc_auc_score(y, p)),
            'pr_auc': float(average_precision_score(y, p)),
            'f1': float(f1_score(y, p >= threshold)),
            'lift_at_10': float(np.asarray(y)[order].mean() / np.asarray(y).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=314)
    args = parser.parse_args()
    splits = {}
    for name in ['train', 'validation', 'test']:
        files = sorted(Path(os.environ['SM_CHANNEL_' + name.upper()]).glob('*.parquet'))
        if not files:
            raise ValueError('No parquet files in ' + name)
        splits[name] = separate(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    X, y = splits['train']
    model, numeric, categorical = build_pipeline(X, args.n_estimators)
    model.fit(X, y)
    xv, yv = splits['validation']
    pv = model.predict_proba(xv[X.columns])[:, 1]
    precision, recall, thresholds = precision_recall_curve(yv, pv)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    xt, yt = splits['test']
    result = {'validation': metrics(yv, pv, threshold),
              'test': metrics(yt, model.predict_proba(xt[X.columns])[:, 1], threshold),
              'threshold': threshold, 'rows': {k: len(v[1]) for k, v in splits.items()}}
    out = Path(os.environ['SM_MODEL_DIR']); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': model, 'columns': X.columns.tolist(), 'numeric': numeric,
                 'categorical': categorical, 'threshold': threshold}, out / 'model.joblib')
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2))
    # Build a schema-aware serving fixture using the exact fitted model.
    sample = xt.head(8)
    from inference import predict_fn
    fixture = {'request': {'instances': json.loads(sample.to_json(orient='records'))},
               'expected': predict_fn(sample.copy(), joblib.load(out / 'model.joblib'))}
    (out / 'smoke.json').write_text(json.dumps(fixture))
    code = out / 'code'; code.mkdir(exist_ok=True)
    for name in ['inference.py', 'requirements.txt']:
        shutil.copy(Path(__file__).parent / name, code / name)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
