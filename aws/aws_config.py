"""
aws_config.py - AWS Configuration Manager
============================================
Centralizes AWS-specific configuration and credential validation.

Usage:
    from aws.aws_config import AWSConfig
    aws_cfg = AWSConfig()
    if aws_cfg.validate_credentials():
        print("AWS is configured!")
"""

import os
from typing import Optional, Dict, Tuple
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class AWSConfig:
    """
    Manages AWS configuration, credential validation, and region settings.
    
    Configuration sources (in priority order):
      1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
      2. AWS credentials file (~/.aws/credentials)
      3. IAM role (for EC2 instances)
      4. config.yaml (aws section)
    """

    def __init__(self):
        # Load from config.yaml
        self.region = cfg("aws.region", "us-east-1")
        self.dry_run = cfg("aws.dry_run", True)
        self.instance_type = cfg("aws.instance_type", "t2.micro")
        self.ami_id = cfg("aws.ami_id", "ami-0c02fb55956c7d316")
        self.max_instances = cfg("aws.max_instances", 10)
        self.min_instances = cfg("aws.min_instances", 2)
        
        # Scaling settings
        self.scale_out_threshold = cfg("aws.scaling.scale_out_threshold", 80)
        self.scale_in_threshold = cfg("aws.scaling.scale_in_threshold", 20)
        self.cooldown_seconds = cfg("aws.scaling.cooldown_seconds", 300)
        
        # CloudWatch settings
        self.cloudwatch_period = cfg("aws.cloudwatch.period", 300)
        self.cloudwatch_history = cfg("aws.cloudwatch.history_minutes", 30)
        
        # Credentials status
        self._credentials_valid = None
        self._account_id = None

    def validate_credentials(self) -> Tuple[bool, str]:
        """
        Validate AWS credentials are configured and working.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not BOTO3_AVAILABLE:
            return False, "boto3 not installed. Run: pip install boto3"
        
        if self.dry_run:
            return True, "Dry-run mode enabled (no credentials needed)"
        
        try:
            sts = boto3.client("sts", region_name=self.region)
            identity = sts.get_caller_identity()
            self._account_id = identity.get("Account")
            self._credentials_valid = True
            return True, f"Credentials valid. Account: {self._account_id}"
        except NoCredentialsError:
            self._credentials_valid = False
            return False, "No AWS credentials found. Configure with: aws configure"
        except ClientError as e:
            self._credentials_valid = False
            return False, f"Credentials error: {e}"

    def get_account_id(self) -> Optional[str]:
        """Get the AWS account ID (requires valid credentials)."""
        if self._account_id is None:
            self.validate_credentials()
        return self._account_id

    def is_production_ready(self) -> Tuple[bool, Dict]:
        """
        Check if all AWS settings are configured for production.
        
        Returns:
            Tuple of (is_ready, issues_dict)
        """
        issues = {}
        
        # Check boto3
        if not BOTO3_AVAILABLE:
            issues["boto3"] = "Not installed"
        
        # Check credentials (skip in dry-run)
        if not self.dry_run:
            valid, msg = self.validate_credentials()
            if not valid:
                issues["credentials"] = msg
        
        # Check required config
        if not self.ami_id or self.ami_id == "ami-0c02fb55956c7d316":
            issues["ami_id"] = "Using default AMI. Set a proper AMI for your region."
        
        # Check security groups
        security_groups = cfg("aws.security_group_ids", [])
        if not security_groups:
            issues["security_groups"] = "No security groups configured"
        
        # Check subnet
        subnet = cfg("aws.subnet_id", None)
        if not subnet:
            issues["subnet"] = "No subnet configured (will use default VPC)"
        
        is_ready = len(issues) == 0 or (self.dry_run and "credentials" not in issues)
        return is_ready, issues

    def print_configuration(self):
        """Print current AWS configuration."""
        print("\n" + "=" * 60)
        print("  AWS CONFIGURATION")
        print("=" * 60)
        print(f"  Region:          {self.region}")
        print(f"  Dry-run mode:    {self.dry_run}")
        print(f"  Instance type:   {self.instance_type}")
        print(f"  AMI ID:          {self.ami_id}")
        print(f"  Max instances:   {self.max_instances}")
        print(f"  Min instances:   {self.min_instances}")
        print(f"  Scale-out at:    {self.scale_out_threshold}% CPU")
        print(f"  Scale-in at:     {self.scale_in_threshold}% CPU")
        print(f"  Cooldown:        {self.cooldown_seconds}s")
        print()
        
        # Validate credentials
        valid, msg = self.validate_credentials()
        status = "✓" if valid else "✗"
        print(f"  Credentials:     {status} {msg}")
        
        # Production readiness
        is_ready, issues = self.is_production_ready()
        status = "✓ Ready" if is_ready else "✗ Not ready"
        print(f"  Production:      {status}")
        if issues:
            for key, issue in issues.items():
                print(f"                   - {key}: {issue}")

    def get_env_vars_template(self) -> str:
        """Return template for environment variables setup."""
        return """
# AWS Credentials Setup
# Add these to your environment or .env file:

# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_DEFAULT_REGION="us-east-1"

# Option 2: AWS CLI (recommended)
# Run: aws configure

# Option 3: IAM Role (for EC2 instances)
# Attach an IAM role with required permissions to your EC2 instance
"""

    def get_required_iam_policy(self) -> Dict:
        """Return the required IAM policy for the ML migration system."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "EC2Operations",
                    "Effect": "Allow",
                    "Action": [
                        "ec2:RunInstances",
                        "ec2:TerminateInstances",
                        "ec2:StopInstances",
                        "ec2:StartInstances",
                        "ec2:DescribeInstances",
                        "ec2:DescribeRegions",
                        "ec2:CreateTags"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "CloudWatchMetrics",
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:GetMetricStatistics",
                        "cloudwatch:ListMetrics",
                        "cloudwatch:GetMetricData"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "STSIdentity",
                    "Effect": "Allow",
                    "Action": [
                        "sts:GetCallerIdentity"
                    ],
                    "Resource": "*"
                }
            ]
        }


def setup_aws_environment():
    """Interactive setup helper for AWS environment."""
    import json
    
    print("\n" + "=" * 60)
    print("  AWS ENVIRONMENT SETUP")
    print("=" * 60)
    
    aws_cfg = AWSConfig()
    
    # Check current status
    print("\n1. Checking current configuration...")
    aws_cfg.print_configuration()
    
    # Print env vars template
    print("\n2. Environment Variables Template:")
    print(aws_cfg.get_env_vars_template())
    
    # Print IAM policy
    print("\n3. Required IAM Policy:")
    print(json.dumps(aws_cfg.get_required_iam_policy(), indent=2))
    
    print("\n" + "=" * 60)
    print("  Setup complete! Update config.yaml with your settings.")
    print("=" * 60)


if __name__ == "__main__":
    setup_aws_environment()
