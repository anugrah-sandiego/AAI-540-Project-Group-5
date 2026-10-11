
import json
import joblib
import os
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Load the model from model_dir."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == 'text/csv':
        # Parse CSV input
        data = request_body.split(',')
        return np.array([float(x) for x in data])
    elif request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    # Reshape if needed
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    # Get probability predictions
    proba = model.predict_proba(input_data)[:, 1]
    return proba


def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'text/csv':
        return ','.join([str(x) for x in prediction])
    elif accept == 'application/json':
        return json.dumps({'prediction': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
