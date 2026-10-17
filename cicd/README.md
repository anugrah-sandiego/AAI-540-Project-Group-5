# Group 5: CI/CD scripts and runbook

Team: Anugrah Rastogi, Dhrub Satyam, Aishwarya Gulhane.
Repository: https://github.com/anugrah-sandiego/AAI-540-Project-Group-5
Based on inspected commit b815d428d20112043e4ab6ecdbb1bd0ece527389 (17 October 2026).

## Install this bundle

Extract the ZIP into the repository root. The `.github` directory is hidden on some systems.
This REPLACES `.github/workflows/ci-cd.yml`; it does not publish Docker Hub images.
Other new files live under `cicd/` and `tests/cicd/`.
The bundle also fixes one existing error: missing `import time` in
`src/aws/sagemaker/batch_inference.py`, which otherwise fails the fatal-error lint check.
Review `group5-cicd.patch` against the stated base commit before applying if your code has changed.
Apply either the ZIP files OR the patch; do not apply both.

The original seven CLI scripts remain available. This release workflow uses a separate
managed training/serving implementation because the old managed classifier artifact does
not contain a fitted encoder. Existing historical model.tar.gz files are NOT migrated or
reused by this workflow. The release trains a new compatible artifact. The local model
comparison pipeline is tested but is not rerun as a prerequisite for every release.

## 1. Run CI without AWS

Use a fresh Python 3.10 virtual environment, rather than upgrading the shared Studio kernel:

```bash
python3.10 -m venv .venv-ci
source .venv-ci/bin/activate
python -m pip install -r cicd/requirements-ci.txt
python -m pip check
ruff check src cicd --select E9,F63,F7,F82
python -m pytest tests -v --junitxml=artifacts/tests.xml --cov=src --cov=cicd --cov-report=xml:artifacts/coverage.xml
```

Push the files to a feature branch and open a PR to main. CI installs dependencies,
checks fatal lint errors, runs existing tests plus new gate/serving/AWS boundary tests,
and saves JUnit and coverage artifacts. No AWS credentials are required for these tests.
New synthetic model fitting uses only tiny test data, not full project training.
Dependencies are separated from the old requirements.txt, whose SDK and AWS CLI pins
are unnecessary for this boto3 release runner. mlflow-skinny supplies the legacy imports
used in model tests. It is not a replacement for a full local MLflow tracking deployment.
CI runtime and managed-container dependencies are intentionally different. No pickle
is loaded across these environments: serving smoke expectations are JSON.

## 2. Demonstrate the success AND failed DAGs (no AWS)

After merging the workflow to the default branch:
1. GitHub repository > Actions > `Group 5 DAG demo (no AWS)` > Run workflow.
2. Choose `success`. Validation, test and simulated deployment succeed; failure handler skips.
3. Run again and choose `failure`. The test exits 1; simulated deployment skips; failure
   handler runs. The overall workflow remains FAILED despite its summary completing.
4. Save screenshots of the Actions graph and logs for your assignment.

These are explicitly simulated deployment states, not claims that AWS deployed or rolled back.
For a real CI failure, temporarily change a test assertion on a disposable branch, push it,
observe the red checks job, then revert the change. Do not weaken production quality gates
just to manufacture a passing model.

## 3. Configure AWS release access

Repository > Settings > Secrets and variables > Actions > Repository secrets:

| Secret | Value |
|---|---|
| AWS_ACCESS_KEY_ID | Active Learner Lab temporary access key |
| AWS_SECRET_ACCESS_KEY | Matching temporary secret key |
| AWS_SESSION_TOKEN | Matching temporary session token (required for lab sessions) |

Do not use a GitHub PAT as an AWS credential. Do not commit `.env` or credentials.
Refresh all three together when the lab session expires, including before approving deployment.
A GitHub runner does not inherit Studio's role/session automatically. Lab policies may
block external runners or some SageMaker operations. The workflow reports these failures;
it does not bypass them. OIDC is preferable outside this restricted lab, but requires an
administrator-configured trust role and is not assumed to be available in Learner Lab.

Create environments under repository Settings > Environments:
- `lab`: allow deployments only from main.
- `production`: allow only main; configure Required reviewers; disable administrator bypass
  if appropriate. Do not rely on the YAML environment name alone to create an approval gate.
  Required reviewers availability depends on repository visibility and GitHub plan.

The protected job explicitly marks the candidate package Approved after review. Rejection
prevents that job from running and leaves the package pending; reviewers can separately
mark it Rejected in SageMaker. No automatic approval occurs on ordinary pushes.

Execution role: arn:aws:iam::562792440372:role/LabRole.
Region: us-east-1. Bucket: sagemaker-us-east-1-562792440372.
The caller needs S3 read/write; SageMaker describe/create training and package APIs;
create/update/delete models, configs and endpoints; invoke endpoint; IAM PassRole for
LabRole; CloudWatch PutMetricAlarm and PutDashboard. The training/serving role also needs
access to its S3 inputs, outputs, source archive, capture location, ECR image and logs.
Do not modify the restricted LabRole policies blindly; ask your course administrator if denied.

## 4. Before the first AWS release

The bucket and this raw CSV object must already exist:
`s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-full.csv`

```bash
aws s3api head-object --bucket sagemaker-us-east-1-562792440372 --key aai540/bank-marketing/raw/bank-full.csv --region us-east-1
```

If missing, upload the raw CSV with the existing uploader or `aws s3 cp` first.
The new prepare stage reads this CSV FROM S3, validates it using the repository schema,
rejects duplicate rows, creates reproducible 70/15/15 stratified splits, then writes all
three Parquet objects to a unique run prefix. It does not need existing processed files.
The raw file's SHA256 and commit are recorded in the release manifest.

The workflow obtains the scikit-learn 1.2-1 framework image from the completed reference job
`bank-marketing-training-20261011-095612`. If that job metadata is unavailable, set
`TRAINING_IMAGE_URI` in the workflow to a verified SageMaker scikit-learn 1.2-1 CPU Python 3
framework image for us-east-1, or change BASE_TRAINING_JOB_NAME to an equivalent job.
Do not point this setting at an unrelated XGBoost image. Training uses framework toolkit
hyperparameters to fetch and run `sourcedir.tar.gz`. Validate container availability in your lab.

## 5. Run the actual release

Actions > `Group 5 CI and SageMaker release` > Run workflow > select main > check release.
Normal push/PR events and manual runs with release unchecked only run CI.

Graph: checks -> train (prepare + managed training) -> evaluate -> register ->
protected deploy (approve + staging + production) -> outcome.
The outcome job runs even on failures and lists actual upstream job results.
All AWS jobs save their latest `artifacts/release.json` as Actions artifacts, including
failed evaluations. Download those artifacts to diagnose the actual stage; do not infer
success from a later summary job's green status.

Training fits an imputer and one-hot encoder on TRAIN only, together with a balanced
Random Forest. It drops duration and target y. Validation chooses an F1 threshold; the
held-out test is evaluated once for this candidate. The model bundle includes fitted
preprocessing, required raw column names, numeric/categorical types and threshold.
No local tuned model or old registry version is silently substituted.
This release implementation does not reproduce the local four-model comparison or its
domain features. It is a separately evaluated candidate, not the historical local winner.

Evaluation requires test ROC-AUC >= 0.75 and top-decile lift >= 2.5. Both finite-value
checks and failures block registration. These defaults come from the project gates;
review and document them before changing. Repeated use of a fixed test set for many
candidate decisions can overfit selection; use an independent temporal acceptance set
for real production governance. The test result is not a guarantee of business impact.

Registration records the exact new artifact with PendingManualApproval. After the
protected approval, staging deploys that package, sends eight held-out raw records,
and compares probabilities and class labels to the exact saved model (atol 1e-6,
rtol 1e-5). Failed schema/parity checks block production. Expected values are generated
in the training environment and transferred as JSON; the CI runner never loads the pickle.
The fixture is not an independent correctness oracle: extend it with business acceptance
cases and load/security tests before real production use.

Production uses a NEW stable endpoint `bank-marketing-production` by default. It does
not touch the historical endpoint `bank-marketing-endpoint-20261011-101615`.
To update the historical endpoint, deliberately change PRODUCTION_ENDPOINT after review.
The first production release creates an endpoint; there is no existing fleet for canary
or rollback. Subsequent releases snapshot the exact prior config and request a SageMaker
blue/green update with a 50%-capacity canary, 300s bake and CloudWatch error alarm.
Production requests TWO ml.m5.xlarge instances so 50% canary capacity is meaningful;
staging requests ONE. Check quota and budget before the AWS run. Canary is capacity-based;
real representative traffic/load must be supplied during baking. The final smoke test
alone does not establish canary safety under load. An idle endpoint has no error samples.

The existing production config is retained for recovery. On post-update smoke failure,
the runner restores that exact config if the endpoint is stable. If SageMaker is still
Updating/Failed, the manifest records operator review rather than falsely claiming a
successful rollback. SageMaker also receives an alarm-based auto-rollback configuration.
A failed FIRST deployment is deleted because there is no previous production config.
Staging cleanup runs even when deployment steps fail; cleanup failure is visible.
The workflow does not send emails or create issues; GitHub run status and summary are
its built-in failure reporting mechanism.

## 6. AWS paths and output artifacts

For release `gh-<run-id>-<attempt>`:

| Resource | Location/name |
|---|---|
| Raw input | s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-full.csv |
| Run root | s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/cicd/gh-<run-id>-<attempt>/ |
| Splits | <run-root>/processed/train.parquet, validation.parquet, test.parquet |
| Source archive | <run-root>/source/sourcedir.tar.gz |
| Model | <run-root>/models/bank-ci-gh-<run-id>-<attempt>/output/model.tar.gz |
| Model archive contents | model.joblib, evaluation.json, smoke.json, code/inference.py, code/requirements.txt |
| Registry group | bank-marketing-models; new version ARN saved in release.json |
| Staging endpoint/config/model | bank-stage-gh-<run-id>-<attempt> (cleaned after deploy) |
| Production endpoint | bank-marketing-production |
| Candidate production config/model | bank-prod-gh-<run-id>-<attempt> |
| Data capture | <run-root>/capture/<endpoint>/AllTraffic/... |
| Endpoint log group | /aws/sagemaker/Endpoints/bank-marketing-production |
| Dashboard | bank-marketing-production |
| Alarm | bank-marketing-production-5xx |

Monitoring here means request/response capture, CloudWatch invocation/error/latency
metrics, dashboard and error alarm. It does NOT create a model-quality, data-drift,
bias monitor, ground-truth join, or scheduled retraining. Those require additional
baseline and label workflows. The legacy monitoring.py is not called because the
inspected implementation mixes baseline/schedule APIs and targets an incorrect endpoint.
Feature Store is not recreated: it is an optional parallel store and is not an input
dependency of this demonstrated S3 training workflow.

## 7. Failure handling and cleanup

| Failure | Result |
|---|---|
| Dependencies, lint, tests | CI red; all AWS jobs skipped |
| S3 access/schema/split failure | Training not started |
| Managed training failure | Evaluation, registration, deployment skipped |
| Quality gate rejection | Evidence retained; registry/deploy skipped |
| Approval rejected | Deploy job does not run; production unchanged |
| Staging/prediction mismatch | Production unchanged; staging cleanup attempted |
| Production smoke failure | Restore previous exact config where endpoint is stable |
| AWS auto-rollback | Candidate config check fails; report failed release |
| Updating/Failed endpoint | Explicit operator recovery required; never claim rollback success |
| Workflow cancellation / credentials expire | Inspect AWS for running jobs/endpoints; do not assume cleanup ran |

The hosted production endpoint CONTINUES BILLING after Actions/Studio ends. The historical
endpoint is separate and may also still bill. Delete endpoints deliberately after evidence
collection. This bundle never empties S3 or deletes existing Feature Store groups, old model
packages, or historical endpoint configs. Failed candidate model/config metadata and run
S3 artifacts may remain for inspection. CloudWatch alarms/dashboard/capture storage can
also incur charges. Back up evidence before targeted cleanup.

If interrupted training continues, stop the specific job saved in release.json with
`aws sagemaker stop-training-job --training-job-name <job> --region us-east-1`.
Delete a no-longer-needed production endpoint with
`aws sagemaker delete-endpoint --endpoint-name bank-marketing-production --region us-east-1`.
Do not delete a retained previous config while it is needed for rollback.

## Sources and verification limits

Repository source was cloned and inspected, rather than relying only on older conversation logs.
Official references:
- https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-canary.html
- https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments

Local tests and syntax checks are reported in VALIDATION.md. No AWS training, deployment,
approval, or GitHub workflow was executed by the assistant. Account permissions, quotas,
container installation, temporary credentials, traffic alarms, and full rollback must be
validated in the user's lab before claiming end-to-end success.
