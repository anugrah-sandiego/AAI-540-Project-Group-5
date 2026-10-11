"""
Upload data to S3 for SageMaker training and feature store.
"""

import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataUploader:
    """Upload local data files to S3 for SageMaker processing."""

    def __init__(self):
        """Initialize S3 client and configuration from environment variables."""
        load_dotenv()

        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = f"sagemaker-{self.region}-{self.account_id}"
        self.base_prefix = "aai540/bank-marketing"

        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
        )

        logger.info(f"Initialized S3 uploader for bucket: {self.bucket_name}")

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """
        Upload a single file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 key (path within bucket)

        Returns:
            S3 URI of the uploaded file
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3_uri = f"s3://{self.bucket_name}/{s3_key}"

        logger.info(f"Uploading {local_path} to {s3_uri}")
        self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)

        logger.info(f"Successfully uploaded to {s3_uri}")
        return s3_uri

    def upload_directory(self, local_dir: str, s3_prefix: str) -> dict:
        """
        Upload all files from a local directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for uploaded files

        Returns:
            Dictionary mapping local filenames to S3 URIs
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        uploaded_files = {}

        for file_path in local_dir.glob("**/*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                s3_uri = self.upload_file(str(file_path), s3_key)
                uploaded_files[file_path.name] = s3_uri

        logger.info(f"Uploaded {len(uploaded_files)} files from {local_dir}")
        return uploaded_files

    def upload_bank_data(self, data_dir: str = "data/raw") -> dict:
        """
        Upload bank marketing dataset files to S3.

        Args:
            data_dir: Local data directory

        Returns:
            Dictionary mapping dataset names to S3 URIs
        """
        data_dir = Path(data_dir)
        uploaded = {}

        # Upload main dataset
        bank_full = data_dir / "bank-full.csv"
        if bank_full.exists():
            uploaded['bank_full'] = self.upload_file(
                str(bank_full),
                f"{self.base_prefix}/raw/bank-full.csv"
            )

        # Upload sample dataset
        bank_sample = data_dir / "bank.csv"
        if bank_sample.exists():
            uploaded['bank_sample'] = self.upload_file(
                str(bank_sample),
                f"{self.base_prefix}/raw/bank.csv"
            )

        # Upload feature descriptions
        bank_names = data_dir / "bank-names.txt"
        if bank_names.exists():
            uploaded['bank_names'] = self.upload_file(
                str(bank_names),
                f"{self.base_prefix}/raw/bank-names.txt"
            )

        return uploaded

    def create_processing_directories(self):
        """Create S3 directories for processed data, models, and outputs."""
        directories = [
            f"{self.base_prefix}/processed/",
            f"{self.base_prefix}/models/",
            f"{self.base_prefix}/inference/",
            f"{self.base_prefix}/monitoring/",
        ]

        for directory in directories:
            # S3 doesn't have real directories, but we can create empty objects with trailing /
            self.s3_client.put_object(Bucket=self.bucket_name, Key=directory)
            logger.info(f"Created S3 directory: {directory}")


def main():
    """Main function to upload data to S3."""
    uploader = S3DataUploader()

    # Create processing directories
    uploader.create_processing_directories()

    # Upload bank marketing data
    uploaded_uris = uploader.upload_bank_data()

    print("\n=== Upload Summary ===")
    for name, uri in uploaded_uris.items():
        print(f"{name}: {uri}")


if __name__ == "__main__":
    main()
