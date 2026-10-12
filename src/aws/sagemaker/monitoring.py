"""
SageMaker Model Monitor and CloudWatch dashboard implementation.
"""

import os
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from dotenv import load_dotenv
import logging
from src.utils.logging_config import setup_logging
from datetime import datetime
import json

# Logging is configured once by setup_logging() when this file is run directly
logger = logging.getLogger(__name__)


class SageMakerModelMonitor:
    """Manage SageMaker Model Monitor and CloudWatch dashboards."""

    def __init__(self):
        """Initialize SageMaker session and configuration."""
        load_dotenv()

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.role_arn = os.getenv('AWS_ROLE_ARN')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.bucket_name = f"sagemaker-{self.region}-{self.account_id}"
        self.base_prefix = "aai540/bank-marketing"

        self.sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        self.sm_client = boto3.client('sagemaker', region_name=self.region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)

        logger.info(f"Initialized Model Monitor for region: {self.region}")

    def create_baseline(self, baseline_data_s3_uri: str,
                       output_s3_uri: str = None,
                       monitor_name: str = "bank-marketing-monitor"):
        """
        Create a baseline for model monitoring using training/validation data.

        Args:
            baseline_data_s3_uri: S3 URI of baseline data
            output_s3_uri: S3 URI for baseline statistics output
            monitor_name: Name of the monitor

        Returns:
            DefaultModelMonitor object
        """
        if output_s3_uri is None:
            output_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/baseline/"

        logger.info(f"Creating baseline with data from {baseline_data_s3_uri}")

        my_monitor = DefaultModelMonitor(
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=self.sagemaker_session
        )

        # Generate baseline statistics and constraints
        my_monitor.create_monitoring_schedule(
            output_s3_uri=output_s3_uri,
            statistics_s3_uri=f"{output_s3_uri}/statistics/",
            constraints_s3_uri=f"{output_s3_uri}/constraints/",
            schedule_cron_expression=None,  # One-time baseline job
            record_preprocessor_script=None,
            post_processor_script=None,
            monitor_name=monitor_name
        )

        # Run baseline job
        baseline_job = my_monitor.run_baseline(
            baseline_input=baseline_data_s3_uri,
            output_s3_uri=output_s3_uri,
            format_dataset=DatasetFormat.csv(header=True)
        )

        logger.info(f"Baseline job started: {baseline_job}")
        return my_monitor

    def create_monitoring_schedule(self, endpoint_name: str,
                                   baseline_s3_uri: str,
                                   output_s3_uri: str = None,
                                   monitor_name: str = "bank-marketing-monitor",
                                   schedule_cron_expression: str = "cron(0 * ? * * *)"):
        """
        Create a monitoring schedule for an endpoint.

        Args:
            endpoint_name: Name of the endpoint to monitor
            baseline_s3_uri: S3 URI of baseline statistics
            output_s3_uri: S3 URI for monitoring output
            monitor_name: Name of the monitor
            schedule_cron_expression: Cron expression for monitoring schedule

        Returns:
            DefaultModelMonitor object
        """
        if output_s3_uri is None:
            output_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/results/"

        logger.info(f"Creating monitoring schedule for endpoint: {endpoint_name}")

        my_monitor = DefaultModelMonitor(
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=self.sagemaker_session
        )

        my_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_name,
            output_s3_uri=output_s3_uri,
            statistics_s3_uri=f"{baseline_s3_uri}/statistics/",
            constraints_s3_uri=f"{baseline_s3_uri}/constraints/",
            schedule_cron_expression=schedule_cron_expression,
            record_preprocessor_script=None,
            post_processor_script=None,
            monitor_name=monitor_name
        )

        logger.info(f"Monitoring schedule created: {monitor_name}")
        return my_monitor

    def create_cloudwatch_dashboard(self, dashboard_name: str = "bank-marketing-dashboard"):
        """
        Create a CloudWatch dashboard for monitoring the ML system.

        Args:
            dashboard_name: Name of the dashboard

        Returns:
            Dashboard configuration
        """
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "ModelLatency", "EndpointName", f"{self.base_prefix}-endpoint"],
                            [".", "Invocations", ".", ".", {"stat": "Sum"}],
                            [".", "4XXError", ".", ".", {"stat": "Sum"}],
                            [".", "5XXError", ".", ".", {"stat": "Sum"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Endpoint Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "EndpointName", f"{self.base_prefix}-endpoint"],
                            [".", "MemoryUtilization", ".", "."],
                            [".", "DiskUtilization", ".", "."]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Infrastructure Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invoke4XXError", "EndpointName", f"{self.base_prefix}-endpoint"],
                            [".", "Invoke5XXError", ".", "."],
                            [".", "ModelLatency", ".", ".", {"stat": "p95"}]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                        "title": "Error Rates and Latency"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", f"{self.base_prefix}-endpoint", {"stat": "Sum"}],
                            [".", "ModelLatency", ".", ".", {"stat": "Average"}]
                        ],
                        "period": 86400,
                        "stat": "Sum",
                        "region": self.region,
                        "title": "Daily Invocation Count"
                    }
                },
                {
                    "type": "log",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "logGroupName": f"/aws/sagemaker/Endpoints/{self.base_prefix}-endpoint",
                        "region": self.region,
                        "title": "Endpoint Logs"
                    }
                }
            ]
        }

        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info(f"CloudWatch dashboard {dashboard_name} created")
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

        return dashboard_body

    def create_model_quality_monitor(self, endpoint_name: str,
                                     ground_truth_s3_uri: str,
                                     output_s3_uri: str = None,
                                     monitor_name: str = "bank-marketing-quality-monitor"):
        """
        Create a model quality monitor to track prediction accuracy.

        Args:
            endpoint_name: Name of the endpoint
            ground_truth_s3_uri: S3 URI of ground truth data
            output_s3_uri: S3 URI for monitoring output
            monitor_name: Name of the quality monitor

        Returns:
            ModelQualityMonitor object
        """
        if output_s3_uri is None:
            output_s3_uri = f"s3://{self.bucket_name}/{self.base_prefix}/monitoring/quality/"

        logger.info(f"Creating model quality monitor for endpoint: {endpoint_name}")

        quality_monitor = DefaultModelMonitor(
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=self.sagemaker_session
        )

        quality_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_name,
            output_s3_uri=output_s3_uri,
            ground_truth_input=ground_truth_s3_uri,
            schedule_cron_expression="cron(0 * ? * * *)",
            monitor_name=monitor_name
        )

        logger.info(f"Model quality monitor created: {monitor_name}")
        return quality_monitor

    def describe_monitoring_schedule(self, monitor_name: str) -> dict:
        """
        Get information about a monitoring schedule.

        Args:
            monitor_name: Name of the monitoring schedule

        Returns:
            Dictionary with monitoring schedule information
        """
        response = self.sm_client.describe_monitoring_schedule(
            MonitoringScheduleName=monitor_name
        )

        return {
            'monitoring_schedule_name': response['MonitoringScheduleName'],
            'monitoring_schedule_status': response['MonitoringScheduleStatus'],
            'monitoring_type': response.get('MonitoringType'),
            'endpoint_name': response.get('EndpointName'),
            'schedule': response.get('ScheduleConfig')
        }

    def stop_monitoring_schedule(self, monitor_name: str):
        """
        Stop a monitoring schedule.

        Args:
            monitor_name: Name of the monitoring schedule
        """
        logger.info(f"Stopping monitoring schedule: {monitor_name}")
        self.sm_client.stop_monitoring_schedule(MonitoringScheduleName=monitor_name)
        logger.info(f"Monitoring schedule stopped")

    def delete_monitoring_schedule(self, monitor_name: str):
        """
        Delete a monitoring schedule.

        Args:
            monitor_name: Name of the monitoring schedule
        """
        logger.info(f"Deleting monitoring schedule: {monitor_name}")
        self.sm_client.delete_monitoring_schedule(MonitoringScheduleName=monitor_name)
        logger.info(f"Monitoring schedule deleted")


def main():
    """Main function to demonstrate monitoring setup."""
    monitor = SageMakerModelMonitor()

    # Create CloudWatch dashboard
    dashboard_body = monitor.create_cloudwatch_dashboard()

    # Report the dashboard through the logger instead of print
    logger.info("CloudWatch dashboard created")
    logger.info("Dashboard widgets: %d", len(dashboard_body['widgets']))

    # Example: Create monitoring schedule
    # endpoint_name = "bank-marketing-endpoint"
    # baseline_s3_uri = f"s3://{monitor.bucket_name}/{monitor.base_prefix}/monitoring/baseline/"
    # monitor.create_monitoring_schedule(endpoint_name, baseline_s3_uri)


if __name__ == "__main__":
    # Set up shared logging only when run as a script, not on import
    setup_logging()
    main()
