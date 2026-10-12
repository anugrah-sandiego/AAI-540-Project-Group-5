"""
Shared logging configuration.

Library modules only create a module-level logger with
``logging.getLogger(__name__)``; entry points (``__main__`` blocks and CLI
scripts) call ``setup_logging()`` once, so every module shares one format,
level and output stream.

Environment variables:
    LOG_LEVEL:  DEBUG, INFO (default), WARNING, ERROR or CRITICAL
    LOG_FORMAT: 'text' (default, human-readable) or 'json' (one JSON object per
                line, for CloudWatch Logs Insights and other log aggregators)
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

TEXT_FORMAT = '%(asctime)s %(levelname)-8s %(name)s: %(message)s'

# Attributes present on every LogRecord; anything else was passed via `extra=`
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {'message', 'asctime'}


class JsonFormatter(logging.Formatter):
    """Format each record as a single JSON line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'time': self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> logging.Logger:
    """
    Configure the root logger to write to stdout. Call once, from an entry point.

    Logging handlers flush after every record, so progress appears immediately
    even when output is redirected to a file or captured by Docker / SageMaker.
    Calling it again replaces the handler it added instead of duplicating output.

    Args:
        level: Log level name; defaults to the LOG_LEVEL env var, then INFO
        fmt: 'text' or 'json'; defaults to the LOG_FORMAT env var, then 'text'

    Returns:
        The configured root logger
    """
    level = (level or os.getenv('LOG_LEVEL', 'INFO')).upper()
    fmt = (fmt or os.getenv('LOG_FORMAT', 'text')).lower()
    if fmt not in ('text', 'json'):
        raise ValueError(f"Unsupported log format: {fmt} (use 'text' or 'json')")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == 'json' else logging.Formatter(TEXT_FORMAT))
    handler._project_handler = True

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, '_project_handler', False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    return root


@contextmanager
def log_stage(logger: logging.Logger, stage: str):
    """
    Log the start, duration and outcome of a pipeline stage.

    Failures are logged with the elapsed time and re-raised; the traceback is
    left to the entry point so it is logged only once.

    Args:
        logger: Logger to write to
        stage: Human-readable stage name
    """
    logger.info('Starting: %s', stage, extra={'stage': stage, 'event': 'start'})
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error('Failed: %s after %.1fs', stage, elapsed,
                     extra={'stage': stage, 'event': 'failed', 'duration_s': round(elapsed, 2)})
        raise
    elapsed = time.perf_counter() - start
    logger.info('Finished: %s in %.1fs', stage, elapsed,
                extra={'stage': stage, 'event': 'finished', 'duration_s': round(elapsed, 2)})
