"""
SageMaker Feature Store implementation for bank marketing dataset.
"""

import os
import pandas as pd
import boto3
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankMarketingFeatureStore:
    """Manage SageMaker Feature Store for bank marketing data."""

    def __init__(self):
        """Initialize SageMaker session and feature store configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.bucket_name = f"sagemaker-{self.region}-{os.getenv('AWS_ACCOUNT_ID')}"
        self.base_prefix = "aai540/bank-marketing"

        self.sagemaker_session = Session(boto3.session.Session(region_name=self.region))

        # Feature group names
        self.feature_group_name = "bank-marketing-features"
        self.feature_group_name_processed = "bank-marketing-features-processed"

        logger.info(f"Initialized Feature Store for region: {self.region}")

    def _get_feature_definitions(self, df: pd.DataFrame) -> list:
        """
        Generate feature definitions from DataFrame schema.

        Args:
            df: DataFrame to extract schema from

        Returns:
            List of feature definitions for Feature Group
        """
        feature_definitions = []

        for column_name, dtype in df.dtypes.items():
            if dtype == 'object':
                feature_type = 'String'
            elif dtype in ['int64', 'int32']:
                feature_type = 'Integral'
            elif dtype in ['float64', 'float32']:
                feature_type = 'Fractional'
            else:
                feature_type = 'String'

            feature_definitions.append({
                'FeatureName': column_name,
                'FeatureType': feature_type
            })

        return feature_definitions

    def create_feature_group(self, df: pd.DataFrame, feature_group_name: str,
                            record_identifier_name: str = 'customer_id',
                            event_time_feature_name: str = 'event_time') -> FeatureGroup:
        """
        Create a SageMaker Feature Group.

        Args:
            df: Sample DataFrame to define schema
            feature_group_name: Name of the feature group
            record_identifier_name: Name of the record identifier column
            event_time_feature_name: Name of the event time column

        Returns:
            FeatureGroup object
        """
        feature_definitions = self._get_feature_definitions(df)

        feature_group = FeatureGroup(
            name=feature_group_name,
            sagemaker_session=self.sagemaker_session,
            feature_definitions=feature_definitions
        )

        # Check if feature group already exists
        try:
            existing = self.sagemaker_session.describe_feature_group(
                FeatureGroupName=feature_group_name
            )
            logger.info(f"Feature group {feature_group_name} already exists")
            return feature_group
        except Exception:
            pass

        # Create feature group
        logger.info(f"Creating feature group: {feature_group_name}")
        feature_group.create(
            s3_uri=f"s3://{self.bucket_name}/{self.base_prefix}/feature-store/",
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=self.role_arn,
            enable_online_store=True,
            enable_offline_store=True
        )

        # Wait for feature group to be created
        logger.info("Waiting for feature group to be created...")
        status = feature_group.describe().get('FeatureGroupStatus')
        while status == 'Creating':
            time.sleep(5)
            status = feature_group.describe().get('FeatureGroupStatus')
            logger.info(f"Feature group status: {status}")

        if status != 'Created':
            raise RuntimeError(f"Failed to create feature group. Status: {status}")

        logger.info(f"Feature group {feature_group_name} created successfully")
        return feature_group

    def ingest_data(self, feature_group: FeatureGroup, df: pd.DataFrame,
                    max_workers: int = 2, wait: bool = True) -> None:
        """
        Ingest data into a Feature Group.

        Args:
            feature_group: FeatureGroup object
            df: DataFrame with data to ingest
            max_workers: Number of parallel workers
            wait: Whether to wait for ingestion to complete
        """
        # Add event time if not present
        if 'event_time' not in df.columns:
            df['event_time'] = datetime.now().isoformat()

        # Add record identifier if not present
        if 'customer_id' not in df.columns:
            df['customer_id'] = range(len(df))

        logger.info(f"Ingesting {len(df)} records into feature group...")
        feature_group.ingest(
            data_frame=df,
            max_workers=max_workers,
            wait=wait
        )
        logger.info("Data ingestion completed")

    def create_bank_marketing_feature_groups(self, data_path: str = "data/raw/bank-full.csv"):
        """
        Create feature groups for bank marketing dataset.

        Args:
            data_path: Path to the raw data file

        Returns:
            Dictionary of created feature groups
        """
        # Load data
        df = pd.read_csv(data_path, sep=';')

        # Create raw feature group
        logger.info("Creating raw feature group...")
        raw_fg = self.create_feature_group(
            df,
            self.feature_group_name
        )

        # Ingest raw data
        self.ingest_data(raw_fg, df)

        # Create processed feature group (after preprocessing)
        # This would be done after running the preprocessing pipeline
        logger.info("Raw feature group created and populated")

        return {
            'raw': raw_fg,
            'processed': None  # To be populated after preprocessing
        }

    def get_feature_group(self, feature_group_name: str) -> FeatureGroup:
        """
        Get an existing Feature Group.

        Args:
            feature_group_name: Name of the feature group

        Returns:
            FeatureGroup object
        """
        return FeatureGroup(
            name=feature_group_name,
            sagemaker_session=self.sagemaker_session
        )

    def retrieve_features(self, feature_group_name: str, record_identifiers: list) -> pd.DataFrame:
        """
        Retrieve features from the Feature Store.

        Args:
            feature_group_name: Name of the feature group
            record_identifiers: List of record identifiers to retrieve

        Returns:
            DataFrame with retrieved features
        """
        feature_group = self.get_feature_group(feature_group_name)

        # Build retrieval query
        feature_group_query = feature_group.create_query(
            feature_group_name=feature_group_name
        )

        # Execute query (for specific records)
        # Note: This is a simplified version; in production you'd use more complex queries
        logger.info(f"Retrieving features for {len(record_identifiers)} records")

        return feature_group_query


def main():
    """Main function to create and populate feature groups."""
    feature_store = BankMarketingFeatureStore()

    # Create feature groups and ingest data
    feature_groups = feature_store.create_bank_marketing_feature_groups()

    print("\n=== Feature Groups Created ===")
    for name, fg in feature_groups.items():
        if fg:
            print(f"{name}: {fg.name}")
            print(f"Status: {fg.describe().get('FeatureGroupStatus')}")


if __name__ == "__main__":
    main()
