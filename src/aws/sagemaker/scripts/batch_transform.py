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
        # Parse CSV input (multiple rows)
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            if line:
                data.append([float(x) for x in line.split(',')])
        return np.array(data)
    elif request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    # Get probability predictions
    proba = model.predict_proba(input_data)[:, 1]
    return proba


def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'text/csv':
        # Return CSV format
        lines = [','.join([str(x) for x in row]) for row in prediction]
        return '\n'.join(lines)
    elif accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
