"""
Pipeline step: build training and evaluation data from the raw batches in S3.

Runs as a SageMaker Processing job. Each run rebuilds the dataset from every
accepted raw batch, so re-running never double-counts rows. Batches already in
the checkpoint become training data; batches new in this run become evaluation
data, which neither the current champion nor the new challenger has trained on.
Batches that fail validation are quarantined once and skipped afterwards.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data_ingestion.data_ingestion import load_data, validate_schema
from src.utils.lineage import file_sha256
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

BASE = Path('/opt/ml/processing')
BATCH_FILE = 'part.csv'
SEPARATOR = ';'


def prepare(raw_dir: Path, checkpoint: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Validate, deduplicate and split the raw batches.

    Args:
        raw_dir: One sub-directory per batch, each holding BATCH_FILE
        checkpoint: batch id -> status from earlier runs

    Returns:
        (training rows, evaluation rows, run report, updated checkpoint)
    """
    status = dict(checkpoint)
    frames, new_ids, quarantined = [], [], {}
    for batch_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        batch_id = batch_dir.name
        if checkpoint.get(batch_id, {}).get('status') == 'quarantined':
            continue
        csv = batch_dir / BATCH_FILE
        try:
            frame = load_data(csv, sep=SEPARATOR)
            validate_schema(frame)
        except (OSError, ValueError) as exc:
            logger.error("Quarantining batch %s: %s", batch_id, exc)
            quarantined[batch_id] = str(exc)
            status[batch_id] = {'status': 'quarantined', 'reason': str(exc)}
            continue
        if batch_id not in checkpoint:
            new_ids.append(batch_id)
        status[batch_id] = {'status': 'processed', 'rows': len(frame), 'sha256': file_sha256(csv)}
        frames.append(frame.assign(batch_id=batch_id))

    if not frames:
        raise ValueError(f"No valid batches in {raw_dir}")
    data = pd.concat(frames, ignore_index=True)
    columns = [column for column in data.columns if column != 'batch_id']
    deduplicated = data.drop_duplicates(subset=columns)

    # On the first run every batch is new: hold out only the newest one for evaluation
    accepted = [frame['batch_id'].iat[0] for frame in frames]
    eval_ids = set(new_ids) if len(new_ids) < len(accepted) else {accepted[-1]}
    is_eval = deduplicated['batch_id'].isin(eval_ids)
    train = deduplicated[~is_eval].drop(columns='batch_id')
    evaluation = deduplicated[is_eval].drop(columns='batch_id')

    report = {'train_rows': len(train), 'eval_rows': len(evaluation),
              'new_batches': new_ids, 'eval_batches': sorted(eval_ids),
              'duplicates_dropped': len(data) - len(deduplicated), 'quarantined': quarantined}
    logger.info("Prepared %d training and %d evaluation rows from %d batches (%d new, %d quarantined)",
                len(train), len(evaluation), len(accepted), len(new_ids), len(quarantined))
    return train, evaluation, report, status


def main() -> None:
    setup_logging()
    checkpoint_file = BASE / 'checkpoint-in' / 'processed_batches.json'
    checkpoint = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {}
    train, evaluation, report, status = prepare(BASE / 'raw', checkpoint)

    for name, frame in (('train', train), ('eval', evaluation)):
        (BASE / name).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(BASE / name / f'{name}.parquet', index=False)
    for batch_id, reason in report['quarantined'].items():
        target = BASE / 'quarantine' / batch_id
        target.mkdir(parents=True, exist_ok=True)
        source = BASE / 'raw' / batch_id / BATCH_FILE
        if source.exists():
            shutil.copy2(source, target / BATCH_FILE)
        (target / 'reason.txt').write_text(reason)

    (BASE / 'report').mkdir(parents=True, exist_ok=True)
    (BASE / 'report' / 'report.json').write_text(json.dumps(report, indent=2))
    (BASE / 'checkpoint').mkdir(parents=True, exist_ok=True)
    (BASE / 'checkpoint' / 'processed_batches.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
