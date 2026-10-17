
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(Path(model_dir) / "model.joblib")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Expected a classifier with predict_proba; incompatible artifact.")
    return model


def input_fn(request_body, request_content_type):
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    kind = request_content_type.split(";")[0].strip().lower()
    if kind == "text/csv":
        rows = list(csv.reader(io.StringIO(request_body)))
        if not rows or not rows[0]:
            raise ValueError("CSV must contain numeric rows without a header.")
        frame = pd.DataFrame(rows)
    elif kind == "application/json":
        data = json.loads(request_body)
        columns = None
        if isinstance(data, dict):
            columns = data.get("columns")
            data = data.get("instances")
        if not isinstance(data, list) or not data:
            raise ValueError("Use a numeric list, a list of rows, or an instances object.")
        if isinstance(data[0], dict):
            if columns is not None:
                raise ValueError("Omit columns when instances are named records.")
            frame = pd.DataFrame(data)
        else:
            if not isinstance(data[0], list):
                data = [data]
            frame = pd.DataFrame(data, columns=columns)
    else:
        raise ValueError("Supported input types: text/csv and application/json.")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Input must be nonempty with unique feature columns.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Use already encoded numeric features; raw categories are unsupported.") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Features must be finite numeric values.")
    return frame


def predict_fn(frame, model):
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and frame.shape[1] != expected_count:
        raise ValueError(f"Expected {expected_count} encoded features; received {frame.shape[1]}.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if isinstance(frame.columns, pd.RangeIndex):
            frame = frame.copy()
            frame.columns = names
        else:
            if set(frame.columns) != set(names):
                raise ValueError("Named columns must exactly match encoded training feature names.")
            frame = frame.loc[:, names]
    elif not isinstance(frame.columns, pd.RangeIndex):
        raise ValueError("Model has no saved feature names; send positional features in training order.")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The model must have positive class label 1.")
    probabilities = model.predict_proba(frame)[:, classes.index(1)]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned nonfinite probabilities.")
    return probabilities


def output_fn(prediction, accept):
    kind = (accept or "application/json").split(";")[0].strip().lower()
    if kind == "text/csv":
        return "\n".join(str(float(value)) for value in prediction)
    if kind in ("application/json", "*/*"):
        return json.dumps({"prediction": prediction.tolist()})
    raise ValueError("Supported response types: text/csv and application/json.")
