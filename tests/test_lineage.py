"""
Tests for the model lineage helpers.
"""

import hashlib
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import pytest
import sklearn
from src.utils import lineage
from src.utils.lineage import (git_info, file_sha256, environment_info, input_schema,
                               collect_lineage, resolve_allow_dirty, code_state,
                               enforce_clean_code, UncommittedCodeError,
                               COMMIT_ENV_VARS, BRANCH_ENV_VARS, ALLOW_DIRTY_ENV_VAR)

requires_git = pytest.mark.skipif(shutil.which('git') is None, reason='git CLI not installed')


@pytest.fixture(autouse=True)
def clear_git_env(monkeypatch):
    """CI sets GITHUB_SHA etc.; clear them (and ALLOW_DIRTY) so tests are deterministic."""
    for name in COMMIT_ENV_VARS + BRANCH_ENV_VARS + (ALLOW_DIRTY_ENV_VAR,):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def git_repo(tmp_path):
    """Throwaway repository on branch 'main' with one committed file."""
    def git(*args):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)

    git('init', '-b', 'main')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    git('config', 'commit.gpgsign', 'false')
    (tmp_path / 'train.py').write_text('print("v1")\n')
    git('add', 'train.py')
    git('commit', '-m', 'initial')
    return tmp_path


# --- git_info ---------------------------------------------------------------

@requires_git
def test_git_info_clean_repo(git_repo):
    info = git_info(git_repo)
    expected = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=git_repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    assert info == {'commit': expected, 'branch': 'main', 'dirty': False,
                    'dirty_files': [], 'source': 'git'}


@requires_git
def test_git_info_detects_modified_and_untracked_files(git_repo):
    (git_repo / 'train.py').write_text('print("v2")\n')
    (git_repo / 'new_feature.py').write_text('x = 1\n')
    info = git_info(git_repo)
    assert info['dirty'] is True
    assert set(info['dirty_files']) == {'train.py', 'new_feature.py'}


@requires_git
def test_git_info_handles_spaces_and_renames(git_repo):
    (git_repo / 'my notes.py').write_text('x = 1\n')
    subprocess.run(['git', 'mv', 'train.py', 'train_v2.py'], cwd=git_repo, check=True, capture_output=True)
    assert set(git_info(git_repo)['dirty_files']) == {'my notes.py', 'train_v2.py'}


@requires_git
def test_git_info_ignores_gitignored_files(git_repo):
    (git_repo / '.gitignore').write_text('*.log\n')
    subprocess.run(['git', 'add', '.gitignore'], cwd=git_repo, check=True, capture_output=True)
    subprocess.run(['git', 'commit', '-m', 'ignore logs'], cwd=git_repo, check=True, capture_output=True)
    (git_repo / 'run.log').write_text('output\n')
    assert git_info(git_repo)['dirty'] is False


@requires_git
def test_git_info_detached_head_has_no_branch(git_repo):
    subprocess.run(['git', 'checkout', '--detach'], cwd=git_repo, check=True, capture_output=True)
    info = git_info(git_repo)
    assert info['branch'] is None and info['source'] == 'git'


def test_git_info_outside_repo_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_SHA', 'abc123')
    monkeypatch.setenv('GITHUB_REF_NAME', 'main')
    info = git_info(tmp_path)
    assert info == {'commit': 'abc123', 'branch': 'main', 'dirty': None,
                    'dirty_files': [], 'source': 'env'}


def test_git_info_without_git_prefers_explicit_env(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage.shutil, 'which', lambda name: None)
    monkeypatch.setenv('GIT_COMMIT', 'from-docker-build-arg')
    monkeypatch.setenv('GITHUB_SHA', 'from-actions')
    assert git_info(tmp_path)['commit'] == 'from-docker-build-arg'


def test_git_info_unavailable(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lineage.shutil, 'which', lambda name: None)
    info = git_info(tmp_path)
    assert info['commit'] is None and info['dirty'] is None
    assert info['source'] == 'unavailable'
    assert 'commit is unknown' in caplog.text


def test_git_info_handles_git_timeout(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd='git', timeout=10)
    monkeypatch.setattr(lineage.subprocess, 'run', timeout)
    monkeypatch.setenv('GIT_COMMIT', 'abc123')
    assert git_info(tmp_path)['source'] == 'env'


# --- file_sha256 -----------------------------------------------------------

def test_file_sha256_matches_hashlib_across_chunks(tmp_path):
    path = tmp_path / 'data.csv'
    content = b'age;job;y\n' * 5000
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert file_sha256(path) == expected
    assert file_sha256(path, chunk_size=7) == expected


def test_file_sha256_changes_when_data_changes(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_bytes(b'age;y\n30;no\n')
    before = file_sha256(path)
    path.write_bytes(b'age;y\n30;yes\n')
    assert file_sha256(path) != before


def test_file_sha256_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / 'missing.csv')


# --- environment_info ---------------------------------------------------------

def test_environment_info_reports_installed_versions():
    info = environment_info()
    assert info['python'].count('.') == 2
    assert info['packages']['scikit-learn'] == sklearn.__version__
    assert info['packages']['pandas'] == pd.__version__
    json.dumps(info)  # must be serialisable for metadata / MLflow


def test_environment_info_missing_package_is_none():
    info = environment_info(packages=['pandas', 'definitely-not-installed-pkg'])
    assert info['packages'] == {'pandas': pd.__version__, 'definitely-not-installed-pkg': None}


# --- collect_lineage --------------------------------------------------------

@requires_git
def test_collect_lineage_combines_git_data_and_environment(git_repo):
    data = git_repo / 'data.csv'
    data.write_bytes(b'age;y\n30;no\n')
    lineage = collect_lineage(data, repo_dir=git_repo)

    assert set(lineage) == {'collected_at', 'git', 'data', 'environment'}
    assert lineage['collected_at'].endswith('+00:00')  # UTC
    assert lineage['git']['source'] == 'git' and lineage['git']['dirty'] is True  # data.csv is untracked
    assert lineage['data'] == {'path': data.as_posix(), 'sha256': file_sha256(data), 'size_bytes': 12}
    assert lineage['environment']['packages']['pandas'] == pd.__version__
    json.dumps(lineage)


def test_collect_lineage_records_project_files_relative_to_repo_root():
    absolute = lineage.PROJECT_ROOT / 'data' / 'raw' / 'bank.csv'
    assert collect_lineage(absolute)['data']['path'] == 'data/raw/bank.csv'


def test_collect_lineage_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_lineage(tmp_path / 'missing.csv', repo_dir=tmp_path)


# --- uncommitted-code policy ------------------------------------------------

CLEAN = {'commit': 'abc123', 'dirty': False, 'dirty_files': [], 'source': 'git'}
DIRTY = {'commit': 'abc123', 'dirty': True, 'dirty_files': ['src/a.py'], 'source': 'git'}
UNKNOWN = {'commit': 'abc123', 'dirty': None, 'dirty_files': [], 'source': 'env'}
NO_COMMIT = {'commit': None, 'dirty': None, 'dirty_files': [], 'source': 'unavailable'}


def test_resolve_allow_dirty_uses_config_when_env_unset():
    assert resolve_allow_dirty(False) is False
    assert resolve_allow_dirty(True) is True


@pytest.mark.parametrize('value, expected', [('1', True), ('TRUE', True), (' yes ', True), ('on', True),
                                             ('0', False), ('false', False), ('No', False), ('off', False)])
def test_resolve_allow_dirty_env_overrides_config_both_ways(monkeypatch, value, expected):
    monkeypatch.setenv(ALLOW_DIRTY_ENV_VAR, value)
    assert resolve_allow_dirty(not expected) is expected


def test_resolve_allow_dirty_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(ALLOW_DIRTY_ENV_VAR, 'maybe')
    with pytest.raises(ValueError, match='ALLOW_DIRTY'):
        resolve_allow_dirty(False)


@pytest.mark.parametrize('git, expected', [(CLEAN, 'clean'), (DIRTY, 'dirty'),
                                           (UNKNOWN, 'unknown'), (NO_COMMIT, 'dirty')])
def test_code_state(git, expected):
    assert code_state(git) == expected


def test_enforce_clean_code_accepts_clean_code(caplog):
    assert enforce_clean_code(CLEAN, allow_dirty=False) == 'clean'
    assert not caplog.records


def test_enforce_clean_code_blocks_uncommitted_changes():
    with pytest.raises(UncommittedCodeError, match=r"uncommitted changes in \['src/a.py'\].*ALLOW_DIRTY=1"):
        enforce_clean_code(DIRTY, allow_dirty=False)


def test_enforce_clean_code_blocks_when_no_commit_is_known():
    with pytest.raises(UncommittedCodeError, match='no git commit could be determined'):
        enforce_clean_code(NO_COMMIT, allow_dirty=False)


def test_enforce_clean_code_allows_dirty_with_override_but_warns(caplog):
    assert enforce_clean_code(DIRTY, allow_dirty=True) == 'dirty'
    assert "allow_dirty is set: uncommitted changes in ['src/a.py']" in caplog.text


def test_enforce_clean_code_warning_names_missing_commit(caplog):
    assert enforce_clean_code(NO_COMMIT, allow_dirty=True) == 'dirty'
    assert 'allow_dirty is set: no git commit could be determined' in caplog.text


def test_enforce_clean_code_allows_unknown_state_with_warning(caplog):
    # e.g. a Docker image without git, built with GIT_COMMIT
    assert enforce_clean_code(UNKNOWN, allow_dirty=False) == 'unknown'
    assert 'Cannot check for uncommitted changes' in caplog.text


# --- input_schema -----------------------------------------------------------

def test_input_schema_columns_order_and_kinds():
    X = pd.DataFrame({
        'age': [30, 41],
        'balance': [1.5, np.nan],
        'job': ['admin.', 'technician'],
        'month': pd.Categorical(['may', 'jun']),
        'is_new': [True, False],
        'contacted_at': pd.to_datetime(['2008-05-01', '2008-06-02']),
    })
    schema = input_schema(X)
    assert [c['name'] for c in schema] == list(X.columns)
    assert {c['name']: c['kind'] for c in schema} == {
        'age': 'numeric', 'balance': 'numeric', 'job': 'categorical',
        'month': 'categorical', 'is_new': 'boolean', 'contacted_at': 'datetime',
    }
    assert {c['name']: c['nullable'] for c in schema}['balance'] is True
    assert {c['name']: c['nullable'] for c in schema}['age'] is False
    json.dumps(schema)


def test_input_schema_text_kind_is_version_independent():
    # pandas 2 stores text as 'object', pandas 3 as 'str'; both must map to 'categorical'
    as_object = pd.DataFrame({'job': pd.Series(['admin.'], dtype=object)})
    as_string = pd.DataFrame({'job': pd.Series(['admin.'], dtype='string')})
    assert input_schema(as_object)[0]['kind'] == input_schema(as_string)[0]['kind'] == 'categorical'
