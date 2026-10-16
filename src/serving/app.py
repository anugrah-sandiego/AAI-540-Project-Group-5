"""
Prediction service for the trained term deposit model.

Serves the SageMaker inference contract (GET /ping, POST /invocations on port
8080) and the same predictions on POST /predict for local clients such as the
Streamlit dashboard, which also reads GET /model to show which model answers.
The model and its metadata are read once from MODEL_DIR: the SageMaker model
directory by default, or a folder filled by `aws-tools pull-model`.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from src.models.predict_model import predict_proba


class PredictRequest(BaseModel):
    """One or more raw customer records, keyed by the model's input columns."""
    instances: List[Dict[str, Any]] = Field(min_length=1)


class Predictor:
    """The trained pipeline plus the decision threshold and input schema saved with it."""

    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / 'model.joblib')
        self.metadata = json.loads((model_dir / 'model_metadata.json').read_text())
        self.schema = self.metadata['input_schema']
        self.threshold = self.metadata['decision_threshold']

    def to_frame(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Order columns as in training and reject missing or non-numeric values."""
        names = [column['name'] for column in self.schema]
        missing = sorted({name for record in records for name in names if name not in record})
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        frame = pd.DataFrame(records)[names]
        for column in self.schema:
            if column['kind'] == 'numeric':
                frame[column['name']] = pd.to_numeric(frame[column['name']])
        return frame

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        probabilities = predict_proba(self.model, self.to_frame(records))
        return [{'probability': round(float(p), 4), 'subscribe': bool(p >= self.threshold)}
                for p in probabilities]

    @property
    def version(self) -> Dict[str, Any]:
        """Identify the model behind every prediction (run, commit, training time)."""
        return {'run_id': self.metadata['run_id'],
                'commit': self.metadata['lineage']['git']['commit'],
                'trained_at': self.metadata['trained_at'],
                'decision_threshold': self.threshold}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(Path(os.getenv('MODEL_DIR', '/opt/ml/model')))


app = FastAPI(title='Term deposit prediction')


@app.get('/ping')
def ping() -> Response:
    """SageMaker health check: healthy once the model has loaded."""
    try:
        get_predictor()
    except (OSError, KeyError, ValueError):
        return Response(status_code=503)
    return Response(status_code=200)


@app.get('/model')
def model_info() -> Dict[str, Any]:
    """Describe the served model: how it was trained, its held-out test metrics and lineage."""
    metadata = get_predictor().metadata
    lineage = metadata['lineage']
    return {**get_predictor().version,
            'model_type': metadata.get('model_type'),
            'refit_on_all_data': metadata.get('refit_on_all_data', False),
            'test_metrics': metadata.get('test_metrics', {}),
            'branch': lineage['git'].get('branch'),
            'data_sha256': lineage.get('data', {}).get('sha256')}


@app.post('/invocations')
@app.post('/predict')
def predict(request: PredictRequest) -> Dict[str, Any]:
    predictor = get_predictor()
    try:
        predictions = predictor.predict(request.instances)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'model': predictor.version, 'predictions': predictions}
