# AWS: yearly training pipeline

The laptop sets things up with temporary AWS Academy keys; AWS then runs on its
own as `LabRole`. Every SageMaker job runs the project's own Docker image, so the
cloud trains and serves exactly the code and package versions tested locally.

```
laptop (Docker, .env keys)            AWS (runs as LabRole)
--------------------------            ---------------------------------------------------------
scripts/deploy.sh      ------------>  ECR: aai540-bank-marketing:<commit>
aws-tools replay       ------------>  S3 raw/ batches
aws-tools deploy-pipeline  -------->  SageMaker Pipeline "bank-marketing-retrain"
aws-tools schedule     ------------>  EventBridge Scheduler  ->  pipeline run:
                                        PrepareData -> EnoughNewData? -> TrainChallenger
                                        -> EvaluateChallenger -> GatesPassed?
                                        -> RegisterChallenger (pending) | RejectChallenger
aws-tools approve / rollback  ----->  Model Registry "bank-marketing-models"
aws-tools pull-model  <------------  champion model.tar.gz  ->  docker compose up api
aws-tools smoke-test  ------------>  serverless endpoint (created, called, deleted)
aws-tools endpoint --up/--down --->  real-time endpoint + data capture + CloudWatch dashboard
aws-tools drift-report  --------->  PSI drift report: captured requests vs training data
aws-tools batch-score  ----------->  Batch Transform job with the champion
aws-tools train-sklearn  --------->  main's training: one random forest in AWS's scikit-learn container
PrepareData (each run)  ---------->  Feature Store "bank-marketing-batches" (new batches, offline store)
GitHub Actions "Deploy to AWS"  -->  scripts/deploy.sh or rollback, run on demand
```

## Code

| File | Runs in | Purpose |
|---|---|---|
| `src/aws/pipeline.py` | aws-tools | Pipeline definition |
| `src/aws/steps/prepare.py` | SageMaker Processing | Validate, quarantine, deduplicate, split raw batches |
| `src/aws/steps/train.py` | SageMaker Training (`train`) | Run the training pipeline in SageMaker's directories |
| `src/aws/steps/evaluate.py` | SageMaker Processing | Challenger vs champion, quality and age-band recall gates |
| `src/serving/app.py` | SageMaker endpoint (`serve`), local `api` | `/ping`, `/invocations`, `/predict` |
| `src/aws/cli.py` | aws-tools | The commands below |
| `src/aws/registry.py` | both | Champion lookup |
| `src/aws/feature_store.py` | aws-tools, SageMaker Processing | Feature group creation (adapted from main) and batch writes |
| `src/aws/endpoint.py` | aws-tools | Real-time endpoint, data capture, CloudWatch dashboard (adapted from main) |
| `src/aws/drift.py` | aws-tools | Drift report on captured requests (replaces Model Monitor, closed to new AWS customers) |
| `src/aws/batch.py` | aws-tools | Batch Transform scoring (adapted from main) |
| `src/aws/sklearn_training.py`, `sklearn/train.py` | aws-tools, SageMaker Training | main's scikit-learn container training (script unchanged from main) |
| `.github/workflows/deploy.yml` | GitHub Actions | Manual deploy or rollback with lab keys from repository secrets |

## S3 layout (`s3://sagemaker-<region>-<account>/aai540/bank-marketing/`)

| Prefix | Contents |
|---|---|
| `raw/batch_id=bNNNN/part.csv` | Incoming batches, never modified |
| `checkpoints/processed_batches.json` | Status, rows and SHA-256 of every batch seen |
| `quarantine/<batch>/` | Batches that failed validation, with `reason.txt` |
| `runs/<execution id>/{train,eval,report,evaluation}/` | Per-run dataset version and decision |
| `models/` | Training job outputs (`model.tar.gz`, MLflow run) |
| `feature-store/` | Offline store of the feature group: every new batch with its arrival time |
| `monitoring/capture/`, `monitoring/reports/` | Captured endpoint traffic and drift reports |
| `batch/<job>/` | Batch Transform input and output |
| `processed/`, `sklearn-models/` | Splits uploaded by `train-sklearn` and its model output |

## How a run decides

1. **PrepareData** rebuilds the dataset from all accepted batches (idempotent). Batches seen before are training data; batches new in this run are evaluation data.
2. **EnoughNewData**: skips training when the new batches have fewer than `MinNewRows` (1,000) rows: enough new customers for a stable evaluation (with 350 rows, ROC-AUC could vary by about ±0.06).
3. **TrainChallenger** runs the training pipeline. It runs the same search as local training (4 models × 3 imbalance strategies, shuffled split, from `config.yaml`); `QuickTraining=true` (`scripts/deploy.sh --quick`) uses logistic regression with a short search for cheaper test runs. Training does not apply the quality gates itself, so a weak model reaches EvaluateChallenger and is rejected with reasons. After choosing the settings and cut-off, it refits on all the history, so the model also learns from the newest data. EvaluateChallenger then judges it on new, later batches with the same gates as local training (ROC-AUC ≥ 0.75, lift@10 ≥ 2.5); on later customers the model scores lower than on the shuffled local test, so cloud runs are expected to end at RejectChallenger.
4. **EvaluateChallenger** scores the challenger and the champion on the new data neither has seen. It passes when the quality gates in `config.yaml` hold, PR-AUC is no more than 0.01 below the champion's, and recall in every age band is at least 80% of overall recall.
5. **GatesPassed** registers the challenger as *pending approval*, or fails the run with the reasons in `evaluation.json`.

## Lab session runbook

Prerequisites: Docker Desktop, the lab started, and `.env` in the project root with
the values from Learner Lab > AWS Details > AWS CLI (never commit or share it):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
```

```bash
docker compose run --rm aws-tools check                      # identity; can the scheduler use LabRole?
docker compose run --rm aws-tools replay --batches 4         # history + 3 batches (1,050 new rows)
scripts/deploy.sh                                            # push this commit's image, update and start the pipeline
# watch the run in SageMaker Studio > Pipelines (about 15-25 minutes)
docker compose run --rm aws-tools approve                    # newest pending -> champion
docker compose run --rm aws-tools smoke-test                 # serverless endpoint, deleted afterwards
docker compose run --rm aws-tools replay --batches 3         # next 1,050 rows before the next run
docker compose run --rm aws-tools schedule --once 2026-10-16T10:00:00   # one run
docker compose run --rm aws-tools schedule --yearly 10-20                # every 20 Oct, 06:00 UTC
docker compose run --rm aws-tools schedule --off                         # remove the schedule
docker compose run --rm aws-tools pull-model && docker compose up api
docker compose run --rm aws-tools rollback                   # if the champion misbehaves
docker compose run --rm aws-tools batch-score                # rank the test split with the champion -> batch_scores.csv
docker compose run --rm aws-tools endpoint --up              # live endpoint + capture + dashboard (bills hourly)
docker compose run --rm aws-tools drift-report               # captured requests vs training data (PSI)
docker compose run --rm aws-tools endpoint --down            # ALWAYS run when finished
docker compose run --rm aws-tools train-sklearn              # main's scikit-learn container training
```

**Cost:** pipeline runs, batch scoring and `train-sklearn` cost cents and leave nothing
running. `endpoint --up` runs an `ml.m5.large` instance (about $0.12 per hour, $2.80 per day)
until `endpoint --down`; the lab's policy denies `ml.t2.medium`. SageMaker Model
Monitor's data-quality jobs are closed to new customers, so `drift-report` compares
the captured requests with the training data instead (PSI; reliable from about 500
captured customers).

**GitHub "Deploy to AWS" button** (Actions tab, `main` only): set the repository secrets
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` to fresh lab keys,
then run the workflow with `deploy` (scripts/deploy.sh) or `rollback`.

End the lab and empty `.env` when done. Cost per run is a few cents (ml.m5.large
jobs for minutes); nothing is left running.

## Limits in AWS Academy

- Keys expire about every 4 hours; only aws-tools uses them. Scheduled runs use `LabRole`.
- If `check` reports that the scheduler cannot use `LabRole`, start runs with `deploy-pipeline --start` instead.
- The lab account closes at the end of the course, so schedules stop then.
- CI (ci-cd.yml) has no AWS access. The manual Deploy workflow uses lab keys stored as repository secrets, which expire with the lab session; a real account would use GitHub OIDC instead.
