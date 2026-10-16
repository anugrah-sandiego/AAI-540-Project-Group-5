#!/usr/bin/env bash
# Deploy the current commit to AWS: build the runtime image, push it to Amazon
# ECR, point the SageMaker pipeline at it and start a run.
#
# Run from the project root in Git Bash, with the lab credentials in .env:
#   scripts/deploy.sh             # full training run
#   scripts/deploy.sh --quick     # extra options go to `aws-tools deploy-pipeline`
set -euo pipefail

REPOSITORY=aai540-bank-marketing

# Every image in ECR must be traceable to a commit
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing to deploy: commit your changes first so the image matches a commit." >&2
  exit 1
fi
TAG=$(git rev-parse --short HEAD)

# Load the keys whether .env lines start with `export` (CloudShell output) or not
set -a; . <(tr -d '\r' < .env); set +a
REGION=${AWS_REGION:-us-east-1}
aws() {
  MSYS_NO_PATHCONV=1 docker run --rm -i -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
    -e AWS_REGION="$REGION" amazon/aws-cli "$@"
}

# Also checks that the keys work before anything is built
ACCOUNT=$(aws sts get-caller-identity --query Account --output text | tr -d '\r')
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPOSITORY:$TAG"

aws ecr describe-repositories --repository-names "$REPOSITORY" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPOSITORY" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE >/dev/null

# Tags are immutable, so a commit already in ECR (e.g. a retry after the keys
# expired mid-deploy) is reused rather than pushed again
if aws ecr describe-images --repository-name "$REPOSITORY" --region "$REGION" \
     --image-ids imageTag="$TAG" >/dev/null 2>&1; then
  echo "Image $IMAGE is already in ECR"
else
  # SageMaker needs a plain linux/amd64 image: no attestation manifests, which
  # would turn the push into an image index
  docker build --platform linux/amd64 --provenance=false --sbom=false \
    --target runtime --build-arg RUN_AS=root \
    --build-arg GIT_COMMIT="$(git rev-parse HEAD)" --build-arg GIT_BRANCH="$(git branch --show-current)" \
    -t "$IMAGE" .
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  docker push "$IMAGE"
fi

docker compose run --rm aws-tools deploy-pipeline --image-tag "$TAG" --start "$@"
