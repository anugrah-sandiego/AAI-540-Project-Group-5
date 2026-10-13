"""
Tests that raw data files are checked out byte-identically on every OS.

Data fingerprints in model lineage hash the raw bytes, so a CRLF checkout on
Windows would give a different fingerprint than Linux / CI for the same commit.
"""

import hashlib
import shutil
import subprocess
import pytest
from src.utils.lineage import PROJECT_ROOT, file_sha256

RAW_CSVS = ['data/raw/bank-full.csv', 'data/raw/bank.csv']


@pytest.mark.parametrize('relative_path', RAW_CSVS)
def test_raw_csv_has_no_crlf_line_endings(relative_path):
    assert b'\r\n' not in (PROJECT_ROOT / relative_path).read_bytes()


@pytest.mark.skipif(shutil.which('git') is None, reason='git CLI not installed')
@pytest.mark.parametrize('relative_path', RAW_CSVS)
def test_raw_csv_matches_committed_bytes(relative_path):
    """The working copy hashes the same as the copy stored in git (when it is unmodified)."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, check=False)

    if git('rev-parse', '--git-dir').returncode != 0:
        pytest.skip('not a git checkout')
    if git('diff', '--quiet', 'HEAD', '--', relative_path).returncode != 0:
        pytest.skip(f'{relative_path} has local changes')

    committed = git('show', f'HEAD:{relative_path}').stdout
    assert file_sha256(PROJECT_ROOT / relative_path) == hashlib.sha256(committed).hexdigest()
