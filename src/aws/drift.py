"""
Data drift report for the live endpoint, in place of SageMaker Model Monitor
(its data-quality jobs are no longer offered to new AWS customers).

Reads the requests the endpoint captured to S3 and compares every input column
with the training data using the population stability index (PSI): below 0.1
is stable, 0.1 to 0.2 a moderate shift, above 0.2 significant drift. The report
is printed and saved next to the captured data.
"""

import base64
import json
import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.aws.config import ENDPOINT_NAME, AwsConfig

DRIFT_PSI, SHIFT_PSI = 0.2, 0.1
BINS = 10
# PSI is noisy on small samples, especially for columns with many categories
MIN_RELIABLE_ROWS = 500


def captured_instances(lines: Iterable[str]) -> pd.DataFrame:
    """Customers from SageMaker data-capture lines (one JSON record per captured request)."""
    rows = []
    for line in lines:
        if line.strip():
            request = json.loads(line)['captureData']['endpointInput']
            data = request['data']
            if request.get('encoding') == 'BASE64':
                data = base64.b64decode(data).decode()
            rows += json.loads(data)['instances']
    return pd.DataFrame(rows)


def psi(reference: pd.Series, current: pd.Series) -> float:
    """Population stability index of one column: deciles of the reference, or its categories."""
    if pd.api.types.is_numeric_dtype(reference):
        # Bin edges at the reference deciles (minimum included, so a constant column still
        # separates its value from anything above it)
        deciles = np.unique(np.quantile(reference, np.linspace(0, 1, BINS + 1)))
        edges = np.concatenate([[-np.inf], deciles[:-1] if len(deciles) > 1 else deciles, [np.inf]])
        expected = pd.cut(reference, edges).value_counts(normalize=True, sort=False)
        actual = pd.cut(current, edges).value_counts(normalize=True, sort=False)
    else:
        expected, actual = reference.value_counts(normalize=True), current.value_counts(normalize=True)
        categories = expected.index.union(actual.index)
        expected, actual = expected.reindex(categories, fill_value=0), actual.reindex(categories, fill_value=0)
    expected, actual = expected.clip(lower=1e-4), actual.clip(lower=1e-4)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI and status per input column, most drifted first."""
    scores = pd.Series({column: psi(reference[column], current[column]) for column in current.columns
                        if column in reference.columns}, name='psi')
    status = pd.cut(scores, [-np.inf, SHIFT_PSI, DRIFT_PSI, np.inf], labels=['stable', 'moderate shift', 'drift'])
    return pd.DataFrame({'psi': scores.round(3), 'status': status}).sort_values('psi', ascending=False)


def drift_report(cfg: AwsConfig, aws, reference: pd.DataFrame) -> pd.DataFrame:
    """Compare everything the endpoint has captured with the reference data; save the report to S3."""
    s3 = aws.client('s3')
    prefix = cfg.key('monitoring', 'capture', ENDPOINT_NAME) + '/'
    keys = [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket=cfg.bucket, Prefix=prefix)
            for item in page.get('Contents', [])]
    if not keys:
        raise SystemExit("No captured requests yet; run `aws-tools endpoint --up` first")
    lines = [line for key in keys
             for line in s3.get_object(Bucket=cfg.bucket, Key=key)['Body'].read().decode().splitlines()]
    current = captured_instances(lines)
    table = drift_table(reference, current)
    report_key = cfg.key('monitoring', 'reports', f'drift-{time.strftime("%Y%m%d-%H%M%S")}.json')
    s3.put_object(Bucket=cfg.bucket, Key=report_key, Body=json.dumps(
        {'captured_customers': len(current), 'columns': table.reset_index(names='column').astype(
            {'status': str}).to_dict(orient='records')}, indent=2).encode())
    print(f"Compared {len(current)} captured customers with {len(reference)} training rows; "
          f"report saved to s3://{cfg.bucket}/{report_key}")
    if len(current) < MIN_RELIABLE_ROWS:
        print(f"Note: fewer than {MIN_RELIABLE_ROWS} captured customers, so PSI values are noisy "
              "(columns with many categories, such as job and month, look drifted by chance).")
    return table
