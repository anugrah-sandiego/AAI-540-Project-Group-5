"""
Model lineage helpers.

Collects the facts needed to trace a trained model back to the code, data and
environment that produced it: git state, a fingerprint of the data file,
library versions and the expected input schema. Every function returns plain
JSON-serialisable values so the results can go straight into
model_metadata.json and MLflow tags.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Repository root (src/utils/lineage.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Libraries that affect training results or whether a saved model can be loaded
DEFAULT_PACKAGES = (
    'scikit-learn', 'pandas', 'numpy', 'imbalanced-learn', 'mlflow',
    'scipy', 'joblib', 'cloudpickle',
)

# Environment variables checked when git is unavailable, e.g. inside Docker
# (pass GIT_COMMIT / GIT_BRANCH as build args) or in GitHub Actions
COMMIT_ENV_VARS = ('GIT_COMMIT', 'GITHUB_SHA')
BRANCH_ENV_VARS = ('GIT_BRANCH', 'GITHUB_REF_NAME')

# Environment override for governance.allow_dirty (e.g. ALLOW_DIRTY=1 for experiments)
ALLOW_DIRTY_ENV_VAR = 'ALLOW_DIRTY'
_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'off'}

GIT_TIMEOUT_SECONDS = 10
MAX_DIRTY_FILES = 20
HASH_CHUNK_BYTES = 1024 * 1024


def _run_git(args: Sequence[str], repo_dir: Path) -> str:
    """Run a git command in repo_dir and return its raw stdout."""
    result = subprocess.run(
        ['git', *args], cwd=repo_dir, capture_output=True, text=True,
        timeout=GIT_TIMEOUT_SECONDS, check=True,
    )
    return result.stdout


def _parse_porcelain_z(output: str) -> List[str]:
    """
    Paths from `git status --porcelain -z` output.

    Each entry is 'XY path' terminated by NUL; renames and copies are followed
    by an extra NUL-terminated entry holding the original path, which is skipped.
    -z also disables quoting, so paths with spaces come through unchanged.
    """
    entries = output.split('\0')
    paths, i = [], 0
    while i < len(entries):
        entry = entries[i]
        if len(entry) > 3:
            paths.append(entry[3:])
            if entry[0] in 'RC':
                i += 1  # skip the original path of a rename / copy
        i += 1
    return paths


def _first_env(names: Sequence[str]) -> Optional[str]:
    """Return the first non-empty environment variable among names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def git_info(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Describe the git state of the code that is running.

    Reads the repository with the git CLI. If git is not installed, the
    directory is not a repository, or a git command fails, falls back to
    environment variables (GIT_COMMIT/GIT_BRANCH, then GitHub Actions'
    GITHUB_SHA/GITHUB_REF_NAME). In that case 'dirty' is None because the
    working-tree state cannot be checked.

    Untracked files that are not git-ignored count as uncommitted changes,
    since a new source file can change behaviour just like an edited one.

    Args:
        repo_dir: Directory inside the repository; defaults to the project root

    Returns:
        Dictionary with:
            commit:      full commit SHA, or None if unknown
            branch:      branch name, or None if detached / unknown
            dirty:       True if there are uncommitted changes, None if unknown
            dirty_files: up to MAX_DIRTY_FILES changed paths (empty if clean)
            source:      'git', 'env' or 'unavailable'
    """
    repo_dir = Path(repo_dir) if repo_dir else PROJECT_ROOT

    if shutil.which('git') is None:
        logger.warning("git is not installed; reading commit from environment variables")
    else:
        try:
            commit = _run_git(['rev-parse', 'HEAD'], repo_dir).strip()
            branch = _run_git(['rev-parse', '--abbrev-ref', 'HEAD'], repo_dir).strip()
            # Not stripped: the first status line can start with a significant space
            changed = _parse_porcelain_z(_run_git(
                ['status', '--porcelain', '-z', '--untracked-files=normal'], repo_dir))
            return {
                'commit': commit,
                'branch': None if branch == 'HEAD' else branch,
                'dirty': bool(changed),
                'dirty_files': changed[:MAX_DIRTY_FILES],
                'source': 'git',
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Could not read git state in %s (%s); "
                           "reading commit from environment variables", repo_dir, exc)

    commit = _first_env(COMMIT_ENV_VARS)
    if commit is None:
        logger.warning("Git commit is unknown; lineage will not identify the source code")
    return {
        'commit': commit,
        'branch': _first_env(BRANCH_ENV_VARS),
        'dirty': None,
        'dirty_files': [],
        'source': 'env' if commit else 'unavailable',
    }


def file_sha256(path: Union[str, Path], chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """
    SHA-256 fingerprint of a file, read in chunks so large files fit in memory.

    Any change to the file's bytes changes the fingerprint, so it proves which
    exact data file a model was trained on.

    Args:
        path: File to hash
        chunk_size: Bytes read per chunk

    Returns:
        Hex-encoded SHA-256 digest

    Raises:
        FileNotFoundError: If the file does not exist
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info(packages: Sequence[str] = DEFAULT_PACKAGES) -> Dict[str, Any]:
    """
    Python, platform and library versions of the current environment.

    Args:
        packages: Distribution names to report (as used by pip)

    Returns:
        Dictionary with 'python', 'implementation', 'platform' and 'packages'
        (name -> version, or None if the package is not installed)
    """
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'python': platform.python_version(),
        'implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'packages': versions,
    }


def collect_lineage(data_path: Union[str, Path],
                    repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Gather code, data and environment lineage for one training run.

    Args:
        data_path: Raw data file the model is trained on
        repo_dir: Directory inside the repository; defaults to the project root

    Returns:
        Dictionary with 'collected_at' (UTC, ISO 8601), 'git' (see git_info),
        'data' (path, sha256, size_bytes) and 'environment' (see environment_info)
    """
    path = Path(data_path)
    # Store project files relative to the repo root, so metadata is portable and
    # does not leak local usernames or directory layouts
    try:
        recorded_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        recorded_path = path.as_posix()
    return {
        'collected_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git': git_info(repo_dir),
        'data': {
            'path': recorded_path,
            'sha256': file_sha256(path),
            'size_bytes': path.stat().st_size,
        },
        'environment': environment_info(),
    }


class UncommittedCodeError(RuntimeError):
    """Raised when training would use code that cannot be traced to a commit."""


def resolve_allow_dirty(config_value: bool = False) -> bool:
    """
    Decide whether training from uncommitted code is allowed.

    The ALLOW_DIRTY environment variable, when set, overrides the config value
    in either direction (1/true/yes/on or 0/false/no/off).

    Args:
        config_value: governance.allow_dirty from config.yaml

    Returns:
        True if uncommitted code may be used

    Raises:
        ValueError: If ALLOW_DIRTY has an unrecognised value
    """
    override = os.getenv(ALLOW_DIRTY_ENV_VAR)
    if override is None or not override.strip():
        return bool(config_value)
    value = override.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{ALLOW_DIRTY_ENV_VAR} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, "
                     f"got {override!r}")


def code_state(git: Dict[str, Any]) -> str:
    """
    Classify the code a run uses, from git_info() output.

    Returns:
        'clean'    committed code with a known commit
        'dirty'    uncommitted changes, or no commit could be found at all
        'unknown'  commit known (e.g. from GIT_COMMIT in Docker) but the
                   working tree could not be checked
    """
    if git.get('commit') is None or git.get('dirty') is True:
        return 'dirty'
    if git.get('dirty') is None:
        return 'unknown'
    return 'clean'


def enforce_clean_code(git: Dict[str, Any], allow_dirty: bool) -> str:
    """
    Apply the uncommitted-code policy before training.

    'dirty' code is refused unless allow_dirty is set; 'unknown' code (no git to
    check, e.g. a Docker image built with GIT_COMMIT) is allowed with a warning,
    since its commit is still recorded.

    Args:
        git: Output of git_info()
        allow_dirty: Result of resolve_allow_dirty()

    Returns:
        The code state ('clean', 'dirty' or 'unknown') that was accepted

    Raises:
        UncommittedCodeError: If the code is dirty and allow_dirty is False
    """
    state = code_state(git)
    reason = ('no git commit could be determined' if git.get('commit') is None
              else f"uncommitted changes in {git.get('dirty_files')}")
    if state == 'dirty' and not allow_dirty:
        raise UncommittedCodeError(
            f"Refusing to train: {reason}. Commit your changes so the model can be traced, "
            f"or set {ALLOW_DIRTY_ENV_VAR}=1 for an experiment.")
    if state == 'dirty':
        logger.warning("Training anyway because allow_dirty is set: %s; this model cannot be "
                       "traced to a commit", reason)
    elif state == 'unknown':
        logger.warning("Cannot check for uncommitted changes (git %s); trusting commit %s",
                       git.get('source'), git.get('commit'))
    return state


def _dtype_kind(series: pd.Series) -> str:
    """Version-independent type family; pandas 2 reports text as 'object', pandas 3 as 'str'."""
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if (isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)):
        return 'categorical'
    return 'other'


def input_schema(X: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Describe the columns a model expects, in training order.

    Args:
        X: Training feature frame (raw columns, before preprocessing)

    Returns:
        One entry per column with:
            name:     column name
            dtype:    pandas dtype as recorded at training time
            kind:     'numeric', 'categorical', 'boolean', 'datetime' or 'other'
            nullable: whether the training data contained missing values
    """
    return [
        {
            'name': str(column),
            'dtype': str(X[column].dtype),
            'kind': _dtype_kind(X[column]),
            'nullable': bool(X[column].isna().any()),
        }
        for column in X.columns
    ]
