# S3 Data Locations and Upload Instructions

## AWS Configuration

- **Account ID:** 562792440372
- **Region:** us-east-1
- **S3 Bucket:** sagemaker-us-east-1-562792440372
- **Role ARN:** arn:aws:iam::562792440372:role/LabRole

## Expected S3 Data Locations

Once uploaded, the data will be stored at the following S3 URIs:

### Raw Data Files
- **Full Dataset:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-full.csv`
- **Sample Dataset:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank.csv`
- **Feature Descriptions:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-names.txt`

### Processing Directories (Created During Upload)
- **Processed Data:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/processed/`
- **Models:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/models/`
- **Inference:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/inference/`
- **Monitoring:** `s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/monitoring/`

## How to Upload Data (Run in SageMaker Lab)

Since you're in a SageMaker lab environment, the credentials are automatically handled. Follow these steps **inside the SageMaker lab notebook or terminal**:

### Option 1: Using Python Script

```bash
cd AAI-540-Project-Group-5
python -m src.aws.sagemaker.data_upload
```

### Option 2: Using SageMaker Notebook

If you're in a SageMaker notebook, you can run:

```python
from src.aws.sagemaker.data_upload import S3DataUploader

uploader = S3DataUploader()
uploader.create_processing_directories()
uploaded_uris = uploader.upload_bank_data()

print("Upload Summary:")
for name, uri in uploaded_uris.items():
    print(f"{name}: {uri}")
```

### Option 3: Using AWS CLI (if available in lab)

```bash
# Create bucket directories
aws s3api put-object --bucket sagemaker-us-east-1-562792440372 --key aai540/bank-marketing/processed/
aws s3api put-object --bucket sagemaker-us-east-1-562792440372 --key aai540/bank-marketing/models/
aws s3api put-object --bucket sagemaker-us-east-1-562792440372 --key aai540/bank-marketing/inference/
aws s3api put-object --bucket sagemaker-us-east-1-562792440372 --key aai540/bank-marketing/monitoring/

# Upload data files
aws s3 cp data/raw/bank-full.csv s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-full.csv
aws s3 cp data/raw/bank.csv s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank.csv
aws s3 cp data/raw/bank-names.txt s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/bank-names.txt
```

## Verification

After upload, verify the files are in S3:

```bash
aws s3 ls s3://sagemaker-us-east-1-562792440372/aai540/bank-marketing/raw/
```

Expected output:
```
2024-10-10 18:19:00    4610348 bank-full.csv
2024-10-10 18:19:00     461474 bank.csv
2024-10-10 18:19:00       3864 bank-names.txt
```

## S3 Console Access

You can also view the uploaded data in the AWS S3 console:

1. Go to AWS Console → S3
2. Navigate to bucket: `sagemaker-us-east-1-562792440372`
3. Browse to: `aai540/bank-marketing/raw/`

## Important Notes

1. **Credentials are not needed in SageMaker lab** - They are automatically handled
2. **Local environment needs credentials** - The local machine doesn't have access to lab credentials
3. **Run in SageMaker lab** - All AWS operations should be performed inside the SageMaker lab environment
4. **Bucket may not exist yet** - The bucket will be created automatically on first upload

## Next Steps After Upload

Once data is uploaded to S3, continue with:

1. Create Feature Store
2. Run Training Jobs
3. Register Model
4. Deploy Endpoint
5. Set Up Monitoring

See `DEPLOYMENT_GUIDE.md` for complete deployment instructions.
