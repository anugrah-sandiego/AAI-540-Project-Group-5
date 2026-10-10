# AWS SageMaker Deployment Guide

This guide provides step-by-step instructions for deploying the term deposit prediction model to AWS SageMaker.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **SageMaker Execution Role** with:
   - S3 read/write access
   - SageMaker full access
   - CloudWatch access
   - IAM pass role permission
3. **Python 3.10+** installed locally
4. **Git** for cloning the repository

## Step 1: Configure AWS Credentials

Create a `.env` file in the project root:

```bash
cd AAI-540-Project-Group-5
cat > .env << 'EOF'
AWS_ACCOUNT_ID=your_account_id
AWS_REGION=us-east-1
AWS_ROLE_ARN=arn:aws:iam::your_account_id:role/your_sagemaker_role
SAGEMAKER_OUTPUT_PATH=s3://sagemaker-us-east-1-your_account_id/aai540/monitoring/
AWS_DEFAULT_REGION=us-east-1
EOF
```

Replace the placeholder values with your actual AWS credentials.

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Upload Data to S3

Upload the bank marketing dataset to S3:

```bash
python -m src.aws.sagemaker.data_upload
```

This will:
- Create S3 directories for processed data, models, and outputs
- Upload `bank-full.csv`, `bank.csv`, and `bank-names.txt` to S3
- Print the S3 URIs of uploaded files

## Step 4: Create Feature Store

Create SageMaker Feature Groups for storing features:

```bash
python -m src.aws.sagemaker.feature_store
```

This will:
- Create a Feature Group for raw bank marketing data
- Ingest the dataset into the Feature Store
- Enable both online and offline stores

## Step 5: Run SageMaker Training

Launch a SageMaker training job:

```bash
python -m src.aws.sagemaker.training
```

This will:
- Prepare the training script
- Create a SKLearn estimator
- Launch a training job on SageMaker
- Save the trained model to S3
- Log training metrics

Monitor the training job in the AWS SageMaker console.

## Step 6: Register Model in Model Registry

Register the trained model in the SageMaker Model Registry:

```bash
python -m src.aws.sagemaker.model_registry
```

This will:
- Create a Model Package Group if it doesn't exist
- Register the model with metrics
- Set the initial approval status to "PendingManualApproval"

You can approve the model in the SageMaker console or via the API.

## Step 7: Deploy Model to Endpoint

Deploy the model to a real-time endpoint:

```bash
python -m src.aws.sagemaker.deployment
```

This will:
- Deploy the model to a SageMaker endpoint
- Enable data capture for monitoring
- Return the endpoint name for invocation

Test the endpoint:

```python
from src.aws.sagemaker.deployment import SageMakerDeployment

deployer = SageMakerDeployment()
result = deployer.invoke_endpoint(
    endpoint_name="your-endpoint-name",
    data=[30, 1500, 5, ...],  # Your feature values
    content_type="text/csv"
)
print(result)
```

## Step 8: Set Up Model Monitoring

Configure model monitoring:

```bash
python -m src.aws.sagemaker.monitoring
```

This will:
- Create a baseline using training/validation data
- Set up a monitoring schedule
- Configure data capture
- Create a CloudWatch dashboard for endpoint metrics

## Step 9: Create Infrastructure Monitoring Dashboards

Create comprehensive monitoring dashboards:

```bash
python -m src.aws.sagemaker.infrastructure_monitoring
```

This will create three CloudWatch dashboards:
- `ml-system-monitoring` - Comprehensive system-wide monitoring
- `training-jobs-monitoring` - Training job specific metrics
- `endpoint-monitoring` - Endpoint performance metrics

## Step 10: Run Batch Inference (Optional)

For batch predictions on large datasets:

```bash
python -m src.aws.sagemaker.batch_inference
```

This will:
- Create a transformer from the registered model
- Process input data from S3
- Output predictions to S3
- Monitor job completion

## CI/CD Pipeline

The project includes a GitHub Actions CI/CD pipeline (`.github/workflows/ci-cd.yml`).

### Setup GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to Repository Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_ACCOUNT_ID`
   - `AWS_ROLE_ARN`
   - `DOCKER_USERNAME` (for Docker Hub)
   - `DOCKER_PASSWORD` (for Docker Hub)

### Pipeline Workflow

The CI/CD pipeline:

1. **Test Job** (on every push/PR):
   - Runs unit tests
   - Performs linting
   - Uploads coverage reports

2. **Build Job** (after tests pass):
   - Builds Docker image
   - Pushes to Docker Hub

3. **Deploy Job** (on main branch only):
   - Uploads data to S3
   - Creates Feature Store
   - Runs SageMaker training
   - Registers model
   - Deploys to endpoint
   - Sets up monitoring

4. **Rollback Job** (on deployment failure):
   - Automatically rolls back to previous model version
   - Creates GitHub issue for notification

5. **Batch Inference Job** (manual trigger):
   - Runs batch transform job on demand

### Manual Trigger

To manually trigger batch inference:

1. Go to Actions tab in GitHub
2. Select "MLOps CI/CD Pipeline"
3. Click "Run workflow"
4. Select "workflow_dispatch" trigger

## Monitoring

### CloudWatch Dashboards

Access the dashboards in the AWS CloudWatch console:
- Navigate to CloudWatch → Dashboards
- Select one of the created dashboards

### SageMaker Console

Monitor training jobs, endpoints, and models:
- Navigate to SageMaker in AWS console
- View Training Jobs, Endpoints, Model Registry, and Feature Store

### Endpoint Invocation

Monitor endpoint invocations and latency in the CloudWatch dashboard or SageMaker console.

## Rollback

If a deployment fails or issues are detected:

```bash
python -m src.aws.sagemaker.rollback
```

This will:
- List available endpoints
- Get the latest approved model from the registry
- Rollback the endpoint to the previous model version

## Cleanup

To delete SageMaker resources:

```python
from src.aws.sagemaker.deployment import SageMakerDeployment
from src.aws.sagemaker.monitoring import SageMakerModelMonitor

deployer = SageMakerDeployment()
monitor = SageMakerModelMonitor()

# Delete endpoint
deployer.delete_endpoint("your-endpoint-name")

# Stop monitoring schedules
monitor.stop_monitoring_schedule("bank-marketing-monitor")
monitor.delete_monitoring_schedule("bank-marketing-monitor")
```

## Troubleshooting

### IAM Permissions Error

Ensure your SageMaker execution role has:
- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`
- `CloudWatchFullAccess`
- IAM pass role permission

### Training Job Fails

Check:
- S3 data paths are correct
- Training script has no syntax errors
- Instance type is available in your region
- Role has necessary permissions

### Endpoint Deployment Fails

Check:
- Model S3 URI is correct
- Instance type is available
- Role has necessary permissions
- Model artifact is valid

### Feature Store Creation Fails

Check:
- Role has SageMaker Feature Store permissions
- S3 bucket exists and is accessible
- Feature group name is unique

## Cost Optimization

To minimize costs:

1. **Use Spot Instances** for training jobs
2. **Auto-scale endpoints** based on traffic
3. **Delete unused endpoints** when not in use
4. **Use multi-model endpoints** for multiple models
5. **Set up budget alerts** in AWS Budgets

## Security Best Practices

1. **Never commit** `.env` file to Git
2. **Use IAM roles** instead of access keys when possible
3. **Enable encryption** for S3 buckets
4. **Use VPC endpoints** for SageMaker
5. **Enable audit logging** with CloudTrail
6. **Rotate credentials** regularly

## Support

For issues or questions:
- Check AWS SageMaker documentation
- Review CloudWatch logs
- Check GitHub Actions workflow logs
- Review this guide and README.md
