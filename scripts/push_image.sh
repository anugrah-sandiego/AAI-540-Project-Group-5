#!/usr/bin/env bash
# Build the runtime image from the current commit and push it to Amazon ECR.
#
# Run from the project root in Git Bash, with the lab credentials in .env:
#   scripts/push_image.sh
# Then deploy the pipeline for the printed tag:
#   docker compose run --rm aws-tools deploy-pipeline --image-tag <tag>
set -euo pipefail

REPOSITORY=aai540-bank-marketing

# Every image in ECR must be traceable to a commit
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing to push: commit your changes first so the image matches a commit." >&2
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

ACCOUNT=$(aws sts get-caller-identity --query Account --output text | tr -d '\r')
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

aws ecr describe-repositories --repository-names "$REPOSITORY" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPOSITORY" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE >/dev/null

# SageMaker needs a plain linux/amd64 image: no attestation manifests, which
# would turn the push into an image index
docker build --platform linux/amd64 --provenance=false --sbom=false \
  --target runtime --build-arg RUN_AS=root \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" --build-arg GIT_BRANCH="$(git branch --show-current)" \
  -t "$REGISTRY/$REPOSITORY:$TAG" .
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker push "$REGISTRY/$REPOSITORY:$TAG"

echo "Pushed $REGISTRY/$REPOSITORY:$TAG"
echo "Next: docker compose run --rm aws-tools deploy-pipeline --image-tag $TAG"
