# AWS integration package for VM Migration Project
"""
AWS Module - Real Cloud Integration
====================================
Provides integration with AWS EC2 and CloudWatch for production deployment.

Modules:
    - cloudwatch_metrics: Fetch real CloudWatch metrics (CPU, Network)
    - ec2_scaler: Auto-scaling actions (launch/terminate instances)
    - aws_config: Configuration and credential management
    - live_pipeline: Production pipeline orchestrator
    - ec2_manager: Legacy EC2 management (backward compatibility)

Usage:
    # Run production pipeline
    from aws.live_pipeline import AWSLivePipeline
    pipeline = AWSLivePipeline()
    pipeline.run_continuous()
    
    # Or use individual components
    from aws.cloudwatch_metrics import CloudWatchMetricsCollector
    from aws.ec2_scaler import EC2Scaler
"""

from aws.cloudwatch_metrics import CloudWatchMetricsCollector
from aws.ec2_scaler import EC2Scaler
from aws.aws_config import AWSConfig

__all__ = [
    "CloudWatchMetricsCollector",
    "EC2Scaler", 
    "AWSConfig",
]
