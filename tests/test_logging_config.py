"""
Tests for the shared logging configuration.
"""

import json
import logging
import pytest
from src.utils.logging_config import setup_logging, log_stage


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Remove the handler setup_logging adds so tests do not leak into each other."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_text_format_writes_to_stdout(capsys):
    setup_logging(level='INFO', fmt='text')
    logging.getLogger('src.example').info('hello %s', 'world')
    out = capsys.readouterr().out
    assert 'INFO' in out and 'src.example: hello world' in out


def test_json_format_includes_extra_fields(capsys):
    setup_logging(level='INFO', fmt='json')
    logging.getLogger('src.example').info('stage done', extra={'stage': 'tune', 'duration_s': 1.5})
    record = json.loads(capsys.readouterr().out.strip())
    assert record['message'] == 'stage done'
    assert record['level'] == 'INFO' and record['logger'] == 'src.example'
    assert record['stage'] == 'tune' and record['duration_s'] == 1.5


def test_level_and_format_come_from_environment(monkeypatch, capsys):
    monkeypatch.setenv('LOG_LEVEL', 'warning')
    monkeypatch.setenv('LOG_FORMAT', 'json')
    setup_logging()
    log = logging.getLogger('src.example')
    log.info('hidden')
    log.warning('shown')
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])['message'] == 'shown'


def test_repeated_setup_does_not_duplicate_output(capsys):
    setup_logging(level='INFO', fmt='text')
    setup_logging(level='INFO', fmt='text')
    logging.getLogger('src.example').info('once')
    assert capsys.readouterr().out.count('once') == 1


def test_unknown_format_raises():
    with pytest.raises(ValueError, match='Unsupported log format'):
        setup_logging(fmt='xml')


def test_log_stage_logs_duration_and_reraises(capsys):
    setup_logging(level='INFO', fmt='json')
    log = logging.getLogger('src.example')

    with log_stage(log, 'load data'):
        pass
    with pytest.raises(RuntimeError):
        with log_stage(log, 'train'):
            raise RuntimeError('boom')

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [(e['stage'], e['event']) for e in events] == [
        ('load data', 'start'), ('load data', 'finished'), ('train', 'start'), ('train', 'failed')]
    assert events[1]['duration_s'] >= 0 and events[3]['level'] == 'ERROR'
