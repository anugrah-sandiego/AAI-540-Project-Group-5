"""JSON records in; probabilities and labels out. Fitted preprocessing is persisted."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    return joblib.load(Path(model_dir) / 'model.joblib')


def input_fn(body, content_type):
    if content_type.split(';')[0] != 'application/json':
        raise ValueError('Use application/json: {"instances": [{feature: value, ...}]}')
    data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get('instances'), list) or not data['instances']:
        raise ValueError('instances must be a nonempty list of records')
    if not all(isinstance(row, dict) for row in data['instances']):
        raise ValueError('Each instance must be an object with named features')
    return pd.DataFrame(data['instances'])


def predict_fn(frame, bundle):
    columns = bundle['columns']
    if set(frame.columns) != set(columns):
        raise ValueError('Feature schema differs from trained model')
    for name in bundle['numeric']:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    for name in bundle['categorical']:
        frame[name] = frame[name].astype(str)
    proba = bundle['pipeline'].predict_proba(frame[columns])[:, 1]
    if not np.isfinite(proba).all():
        raise ValueError('Nonfinite probabilities')
    return {'probabilities': proba.tolist(),
            'predictions': (proba >= bundle['threshold']).astype(int).tolist()}


def output_fn(prediction, accept):
    if accept != 'application/json':
        raise ValueError('Use Accept application/json')
    return json.dumps(prediction)
