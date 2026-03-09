"""
ec2_manager.py - AWS EC2 Integration (Enhanced)
==================================================
Uses boto3 to manage EC2 instances and CloudWatch metrics.
Enhancements:
  - Dry-run mode (configurable, default when no credentials)
  - Auto scale-out when overload predicted
  - Automatic idle instance termination
  - Real CloudWatch metric fetching

SETUP: Configure AWS credentials before using:
    export AWS_ACCESS_KEY_ID=your_key
    export AWS_SECRET_ACCESS_KEY=your_secret
    export AWS_DEFAULT_REGION=us-east-1
"""

import os
import sys
import datetime

from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class EC2Manager:
    """
    Manages AWS EC2 instances for real-world VM migration testing.

    Features:
      - Launch / stop / terminate instances
      - Fetch real CloudWatch CPU metrics
      - Auto scale-out when predicted overload
      - Auto-terminate idle instances
      - Dry-run mode for safe local testing
    """

    def __init__(self, region=None):
        self.region = region or cfg("aws.region", "us-east-1")
        self.dry_run = cfg("aws.dry_run", True)
        self.instances = []
        self.max_instances = cfg("aws.max_instances", 5)
        self.instance_type = cfg("aws.instance_type", "t2.micro")
        self.ami_id = cfg("aws.ami_id", "ami-0c02fb55956c7d316")
        self.auto_scale_threshold = cfg("aws.auto_scale_threshold", 85)
        self.idle_terminate_minutes = cfg("aws.auto_terminate_idle_minutes", 30)

        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            logger.warning("AWS features will run in dry-run/simulation mode.")
            self.ec2 = None
            self.cloudwatch = None
            self.dry_run = True
            return

        if self.dry_run:
            logger.info(f"[AWS] Dry-run mode enabled (region={self.region})")
            self.ec2 = None
            self.cloudwatch = None
            return

        try:
            self.ec2 = boto3.client("ec2", region_name=self.region)
            self.cloudwatch = boto3.client("cloudwatch", region_name=self.region)
            self.ec2.describe_regions(RegionNames=[self.region])
            logger.info(f"[AWS] Connected to AWS region: {self.region}")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"[AWS] Credentials not configured: {e}")
            logger.warning("[AWS] Falling back to dry-run mode.")
            self.ec2 = None
            self.cloudwatch = None
            self.dry_run = True

    # ------------------------------------------------------------------ #
    #  Instance management
    # ------------------------------------------------------------------ #
    def launch_instances(self, count=2, instance_type=None, ami_id=None):
        """Launch EC2 instances (or simulate in dry-run)."""
        instance_type = instance_type or self.instance_type
        ami_id = ami_id or self.ami_id

        if self.dry_run or not self.ec2:
            logger.info(f"[DRY-RUN] Simulating launch of {count} x {instance_type}")
            simulated_ids = [f"i-dry-run-{i+1:04d}" for i in range(count)]
            self.instances.extend(simulated_ids)
            logger.info(f"[DRY-RUN] Instances: {simulated_ids}")
            return simulated_ids

        if len(self.instances) + count > self.max_instances:
            logger.warning(f"[AWS] Would exceed max instances ({self.max_instances}). Aborting.")
            return []

        try:
            response = self.ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                MinCount=count,
                MaxCount=count,
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Project", "Value": "VM-Migration-ML"}]
                }]
            )
            instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
            self.instances.extend(instance_ids)
            logger.info(f"[AWS] Launched {count} instances: {instance_ids}")
            return instance_ids
        except ClientError as e:
            logger.error(f"[AWS] Failed to launch instances: {e}")
            return []

    def stop_instance(self, instance_id):
        """Stop an EC2 instance."""
        if self.dry_run or not self.ec2:
            logger.info(f"[DRY-RUN] Simulated stopping: {instance_id}")
            return
        try:
            self.ec2.stop_instances(InstanceIds=[instance_id])
            logger.info(f"[AWS] Stopped instance: {instance_id}")
        except ClientError as e:
            logger.error(f"[AWS] Failed to stop {instance_id}: {e}")

    def terminate_all(self):
        """Terminate all instances launched in this session."""
        if not self.instances:
            logger.info("[AWS] No instances to terminate.")
            return
        if self.dry_run or not self.ec2:
            logger.info(f"[DRY-RUN] Simulated termination of: {self.instances}")
            self.instances = []
            return
        try:
            self.ec2.terminate_instances(InstanceIds=self.instances)
            logger.info(f"[AWS] Terminated: {self.instances}")
            self.instances = []
        except ClientError as e:
            logger.error(f"[AWS] Failed to terminate: {e}")

    # ------------------------------------------------------------------ #
    #  CloudWatch metrics
    # ------------------------------------------------------------------ #
    def get_cpu_metrics(self, instance_id, period=None, minutes=None):
        """Fetch CPU utilization from CloudWatch (or simulate)."""
        period = period or cfg("aws.cloudwatch.period", 300)
        minutes = minutes or cfg("aws.cloudwatch.history_minutes", 30)

        if self.dry_run or not self.cloudwatch:
            import random
            simulated = []
            now = datetime.datetime.utcnow()
            for i in range(minutes // 5):
                simulated.append({
                    "timestamp": (now - datetime.timedelta(minutes=i * 5)).isoformat(),
                    "cpu_percent": round(random.uniform(20, 90), 2),
                })
            logger.info(f"[DRY-RUN] Simulated {len(simulated)} CPU datapoints for {instance_id}")
            return simulated

        try:
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(minutes=minutes)
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=["Average"],
            )
            datapoints = []
            for dp in response.get("Datapoints", []):
                datapoints.append({
                    "timestamp": dp["Timestamp"].isoformat(),
                    "cpu_percent": round(dp["Average"], 2),
                })
            datapoints.sort(key=lambda x: x["timestamp"])
            logger.info(f"[AWS] Fetched {len(datapoints)} datapoints for {instance_id}")
            return datapoints
        except ClientError as e:
            logger.error(f"[AWS] Failed to get metrics for {instance_id}: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Auto scale-out
    # ------------------------------------------------------------------ #
    def auto_scale_out(self, model, scaler, instance_ids):
        """
        Check each instance for predicted overload;
        if overloaded, launch a new instance to absorb load.

        Args:
            model: Trained ML model.
            scaler: Fitted scaler.
            instance_ids (list): Instance IDs to monitor.

        Returns:
            list: Newly launched instance IDs.
        """
        import numpy as np
        new_instances = []

        for iid in instance_ids:
            metrics = self.get_cpu_metrics(iid, minutes=5)
            if not metrics:
                continue

            latest_cpu = metrics[-1]["cpu_percent"]
            features = np.array([[latest_cpu, 50.0, 30.0, latest_cpu, 50.0]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            if prediction == 1 and latest_cpu > self.auto_scale_threshold:
                logger.info(f"[AUTO-SCALE] {iid} overloaded (CPU={latest_cpu}%). Launching new instance...")
                launched = self.launch_instances(count=1)
                new_instances.extend(launched)

        return new_instances

    # ------------------------------------------------------------------ #
    #  Auto termination of idle instances
    # ------------------------------------------------------------------ #
    def auto_terminate_idle(self, instance_ids, idle_threshold=10.0):
        """
        Terminate instances that have been idle (CPU < threshold)
        for longer than the configured duration.

        Args:
            instance_ids (list): Instance IDs to check.
            idle_threshold (float): CPU % below which instance is idle.

        Returns:
            list: Terminated instance IDs.
        """
        terminated = []
        for iid in instance_ids:
            metrics = self.get_cpu_metrics(iid, minutes=self.idle_terminate_minutes)
            if not metrics:
                continue
            avg_cpu = sum(m["cpu_percent"] for m in metrics) / len(metrics)
            if avg_cpu < idle_threshold:
                logger.info(f"[AUTO-TERMINATE] {iid} idle (avg CPU={avg_cpu:.1f}%). Terminating...")
                self.stop_instance(iid)
                terminated.append(iid)

        return terminated

    # ------------------------------------------------------------------ #
    #  ML prediction on real EC2
    # ------------------------------------------------------------------ #
    def apply_ml_model_to_ec2(self, model, scaler, instance_id):
        """Fetch real CPU metrics and predict overload."""
        import numpy as np
        metrics = self.get_cpu_metrics(instance_id, minutes=5)
        if not metrics:
            return False

        latest_cpu = metrics[-1]["cpu_percent"]
        features = np.array([[latest_cpu, 50.0, 30.0, latest_cpu, 50.0]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        status = "OVERLOADED" if prediction == 1 else "Normal"
        logger.info(f"[AWS] {instance_id} -- CPU: {latest_cpu}% -- Prediction: {status}")
        return prediction == 1


def demo_aws_integration():
    """Demonstrate AWS integration (dry-run safe)."""
    logger.info("=" * 60)
    logger.info("AWS EC2 INTEGRATION DEMO")
    logger.info("=" * 60)

    manager = EC2Manager()
    instance_ids = manager.launch_instances(count=3)

    for iid in instance_ids:
        metrics = manager.get_cpu_metrics(iid, minutes=15)
        if metrics:
            avg_cpu = sum(m["cpu_percent"] for m in metrics) / len(metrics)
            logger.info(f"  {iid} -- Avg CPU: {avg_cpu:.1f}%")

    # Auto-terminate idle
    terminated = manager.auto_terminate_idle(instance_ids)
    if terminated:
        logger.info(f"  Auto-terminated idle instances: {terminated}")

    manager.terminate_all()

    logger.info("AWS DEMO COMPLETE")
    logger.info("=" * 60)
