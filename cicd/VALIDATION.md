# Validation record

Date: 17 October 2026.
Repository baseline: b815d428d20112043e4ab6ecdbb1bd0ece527389.

Completed locally:
- `python -m pytest tests -q`: **38 passed**, two protobuf deprecation warnings, 4.84 seconds.
- `ruff check src cicd --select E9,F63,F7,F82`: passed after adding the missing `time` import to the existing batch inference module.
- `python -m pip check`: no broken requirements in the validation environment.
- Both GitHub workflow files parsed as YAML; job dependency names checked.
- Python source compilation passed.
- Additional local training-to-serving check: a stratified subset of 2,500 actual bank-full rows, five trees, separate train/validation/test Parquet channel directories; training produced model.joblib, evaluation.json, smoke.json and code assets; loaded serving bundle returned exactly the saved fixture probabilities and labels.

The validation host used Python 3.12. GitHub Actions is configured for Python 3.10.
The managed SageMaker scikit-learn 1.2-1 image has its own Python/runtime; the container
requirements are separate. The complete image installation, framework script loading,
AWS requests, account permissions, quotas, canary, alarm behavior and approval environment
were NOT executed locally. Test results do not assert those cloud stages succeeded.
No AWS resources or GitHub deployments were created by this work.
