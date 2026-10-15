"""Guard: no AWS account ID or account-specific ARN may be committed."""

import re
import subprocess
import shutil

import pytest

from src.utils.lineage import PROJECT_ROOT

ACCOUNT_SPECIFIC = re.compile(r'arn:aws:iam::\d{12}|\d{12}\.dkr\.ecr|sagemaker-[a-z0-9-]+-\d{12}')


@pytest.mark.skipif(shutil.which('git') is None, reason='git CLI not installed')
def test_tracked_files_contain_no_account_ids():
    files = subprocess.run(['git', 'ls-files'], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
    offenders = [name for name in files.splitlines()
                 if not name.startswith('data/') and (PROJECT_ROOT / name).is_file()
                 and ACCOUNT_SPECIFIC.search((PROJECT_ROOT / name).read_text(errors='ignore'))]
    assert offenders == []
