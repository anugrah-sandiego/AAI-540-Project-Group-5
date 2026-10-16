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

## S3 layout (`s3://sagemaker-<region>-<account>/aai540/bank-marketing/`)

| Prefix | Contents |
|---|---|
| `raw/batch_id=bNNNN/part.csv` | Incoming batches, never modified |
| `checkpoints/processed_batches.json` | Status, rows and SHA-256 of every batch seen |
| `quarantine/<batch>/` | Batches that failed validation, with `reason.txt` |
| `runs/<execution id>/{train,eval,report,evaluation}/` | Per-run dataset version and decision |
| `models/` | Training job outputs (`model.tar.gz`, MLflow run) |

## How a run decides

1. **PrepareData** rebuilds the dataset from all accepted batches (idempotent). Batches seen before are training data; batches new in this run are evaluation data.
2. **EnoughNewData**: skips training when the new batches have fewer than `MinNewRows` (300) rows.
3. **TrainChallenger** runs the training pipeline. It trains the candidates in `config.yaml` (logistic regression) with a time-ordered split; `QuickTraining=true` (`scripts/deploy.sh --quick`) uses a shorter search for cheaper test runs. Training does not apply the quality gates itself, so a weak model reaches EvaluateChallenger and is rejected with reasons.
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
docker compose run --rm aws-tools replay --batches 2         # history + one day
scripts/deploy.sh                                            # push this commit's image, update and start the pipeline
# watch the run in SageMaker Studio > Pipelines (about 15-25 minutes)
docker compose run --rm aws-tools approve                    # newest pending -> champion
docker compose run --rm aws-tools smoke-test                 # serverless endpoint, deleted afterwards
docker compose run --rm aws-tools replay                     # next day
docker compose run --rm aws-tools schedule --once 2026-10-16T10:00:00   # one run
docker compose run --rm aws-tools schedule --yearly 10-20                # every 20 Oct, 06:00 UTC
docker compose run --rm aws-tools schedule --off                         # remove the schedule
docker compose run --rm aws-tools pull-model && docker compose up api
docker compose run --rm aws-tools rollback                   # if the champion misbehaves
```

End the lab and empty `.env` when done. Cost per run is a few cents (ml.m5.large
jobs for minutes); nothing is left running.

## Limits in AWS Academy

- Keys expire about every 4 hours; only aws-tools uses them. Scheduled runs use `LabRole`.
- If `check` reports that the scheduler cannot use `LabRole`, start runs with `deploy-pipeline --start` instead.
- The lab account closes at the end of the course, so schedules stop then.
- GitHub Actions has no AWS access; CI tests and builds, and `scripts/deploy.sh` pushes the image and starts the pipeline.
