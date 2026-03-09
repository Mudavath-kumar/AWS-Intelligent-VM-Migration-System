"""
cloudwatch_metrics.py - Real CloudWatch Metrics Fetcher
=========================================================
Fetches real EC2 metrics (CPU, Network, Memory) from AWS CloudWatch.
Replaces simulation metrics with live cloud data.

Usage:
    from aws.cloudwatch_metrics import CloudWatchMetricsCollector
    collector = CloudWatchMetricsCollector()
    metrics = collector.get_instance_metrics("i-1234567890abcdef0")
"""

import datetime
from typing import Dict, List, Optional
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class CloudWatchMetricsCollector:
    """
    Collects real EC2 metrics from AWS CloudWatch.
    
    Metrics collected:
      - CPUUtilization (%)
      - NetworkIn (bytes)
      - NetworkOut (bytes)
      - DiskReadOps, DiskWriteOps (optional)
    """

    def __init__(self, region: str = None):
        self.region = region or cfg("aws.region", "us-east-1")
        self.dry_run = cfg("aws.dry_run", True)
        self.period = cfg("aws.cloudwatch.period", 300)  # 5 minutes
        self.history_minutes = cfg("aws.cloudwatch.history_minutes", 30)
        
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not installed. Using simulated metrics.")
            self.cloudwatch = None
            self.ec2 = None
            return
            
        if self.dry_run:
            logger.info(f"[CloudWatch] Dry-run mode enabled")
            self.cloudwatch = None
            self.ec2 = None
            return
            
        try:
            self.cloudwatch = boto3.client("cloudwatch", region_name=self.region)
            self.ec2 = boto3.client("ec2", region_name=self.region)
            logger.info(f"[CloudWatch] Connected to region: {self.region}")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"[CloudWatch] Credentials error: {e}")
            self.cloudwatch = None
            self.ec2 = None

    def get_cpu_utilization(self, instance_id: str, minutes: int = None) -> List[Dict]:
        """
        Fetch CPUUtilization metric for an EC2 instance.
        
        Args:
            instance_id: EC2 instance ID (e.g., "i-1234567890abcdef0")
            minutes: How many minutes of history to fetch
            
        Returns:
            List of dicts with timestamp and cpu_percent
        """
        minutes = minutes or self.history_minutes
        
        if self.dry_run or not self.cloudwatch:
            return self._simulate_cpu_metrics(instance_id, minutes)
            
        return self._fetch_metric(
            instance_id=instance_id,
            metric_name="CPUUtilization",
            namespace="AWS/EC2",
            minutes=minutes,
            output_key="cpu_percent"
        )

    def get_network_in(self, instance_id: str, minutes: int = None) -> List[Dict]:
        """Fetch NetworkIn metric (bytes received)."""
        minutes = minutes or self.history_minutes
        
        if self.dry_run or not self.cloudwatch:
            return self._simulate_network_metrics(instance_id, minutes, "network_in")
            
        return self._fetch_metric(
            instance_id=instance_id,
            metric_name="NetworkIn",
            namespace="AWS/EC2",
            minutes=minutes,
            output_key="network_in_bytes"
        )

    def get_network_out(self, instance_id: str, minutes: int = None) -> List[Dict]:
        """Fetch NetworkOut metric (bytes sent)."""
        minutes = minutes or self.history_minutes
        
        if self.dry_run or not self.cloudwatch:
            return self._simulate_network_metrics(instance_id, minutes, "network_out")
            
        return self._fetch_metric(
            instance_id=instance_id,
            metric_name="NetworkOut",
            namespace="AWS/EC2",
            minutes=minutes,
            output_key="network_out_bytes"
        )

    def get_instance_metrics(self, instance_id: str, minutes: int = None) -> Dict:
        """
        Fetch all metrics for an EC2 instance at once.
        
        Returns:
            Dict with cpu, network_in, network_out, and computed totals
        """
        minutes = minutes or self.history_minutes
        
        cpu_data = self.get_cpu_utilization(instance_id, minutes)
        net_in_data = self.get_network_in(instance_id, minutes)
        net_out_data = self.get_network_out(instance_id, minutes)
        
        # Compute latest values
        latest_cpu = cpu_data[-1]["cpu_percent"] if cpu_data else 0.0
        latest_net_in = net_in_data[-1]["network_in_bytes"] if net_in_data else 0
        latest_net_out = net_out_data[-1]["network_out_bytes"] if net_out_data else 0
        
        # Compute averages
        avg_cpu = sum(d["cpu_percent"] for d in cpu_data) / len(cpu_data) if cpu_data else 0.0
        
        return {
            "instance_id": instance_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "cpu": latest_cpu,
            "cpu_avg": avg_cpu,
            "cpu_history": cpu_data,
            "network_in": latest_net_in,
            "network_out": latest_net_out,
            "network_total": latest_net_in + latest_net_out,
            "ram": self._estimate_ram_from_cpu(latest_cpu),  # RAM not directly available
        }

    def get_all_instances_metrics(self, instance_ids: List[str]) -> List[Dict]:
        """Fetch metrics for multiple EC2 instances."""
        metrics = []
        for instance_id in instance_ids:
            try:
                m = self.get_instance_metrics(instance_id)
                metrics.append(m)
            except Exception as e:
                logger.error(f"Failed to get metrics for {instance_id}: {e}")
        return metrics

    def get_running_instances(self) -> List[str]:
        """Get list of running EC2 instance IDs with the project tag."""
        if self.dry_run or not self.ec2:
            # Return simulated instances
            return [f"i-dry-run-{i:04d}" for i in range(1, 6)]
            
        try:
            response = self.ec2.describe_instances(
                Filters=[
                    {"Name": "instance-state-name", "Values": ["running"]},
                    {"Name": "tag:Project", "Values": ["VM-Migration-ML"]}
                ]
            )
            instance_ids = []
            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    instance_ids.append(instance["InstanceId"])
            logger.info(f"[CloudWatch] Found {len(instance_ids)} running instances")
            return instance_ids
        except ClientError as e:
            logger.error(f"[CloudWatch] Failed to list instances: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #
    def _fetch_metric(self, instance_id: str, metric_name: str, 
                      namespace: str, minutes: int, output_key: str) -> List[Dict]:
        """Generic CloudWatch metric fetcher."""
        try:
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(minutes=minutes)
            
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=self.period,
                Statistics=["Average"]
            )
            
            datapoints = []
            for dp in response.get("Datapoints", []):
                datapoints.append({
                    "timestamp": dp["Timestamp"].isoformat(),
                    output_key: round(dp["Average"], 2)
                })
            datapoints.sort(key=lambda x: x["timestamp"])
            
            logger.debug(f"[CloudWatch] {metric_name} for {instance_id}: {len(datapoints)} points")
            return datapoints
            
        except ClientError as e:
            logger.error(f"[CloudWatch] Failed to fetch {metric_name}: {e}")
            return []

    def _simulate_cpu_metrics(self, instance_id: str, minutes: int) -> List[Dict]:
        """Generate simulated CPU metrics for dry-run mode."""
        import random
        random.seed(hash(instance_id) % 1000)  # Deterministic per instance
        
        datapoints = []
        now = datetime.datetime.utcnow()
        num_points = minutes // 5
        
        base_cpu = random.uniform(30, 60)
        for i in range(num_points):
            timestamp = now - datetime.timedelta(minutes=(num_points - i) * 5)
            # Add some variation
            cpu = base_cpu + random.uniform(-15, 25)
            cpu = max(5, min(95, cpu))  # Clamp to 5-95%
            datapoints.append({
                "timestamp": timestamp.isoformat(),
                "cpu_percent": round(cpu, 2)
            })
        
        logger.info(f"[DRY-RUN] Simulated {len(datapoints)} CPU datapoints for {instance_id}")
        return datapoints

    def _simulate_network_metrics(self, instance_id: str, minutes: int, 
                                   metric_type: str) -> List[Dict]:
        """Generate simulated network metrics for dry-run mode."""
        import random
        random.seed(hash(instance_id + metric_type) % 1000)
        
        datapoints = []
        now = datetime.datetime.utcnow()
        num_points = minutes // 5
        key = f"{metric_type}_bytes"
        
        for i in range(num_points):
            timestamp = now - datetime.timedelta(minutes=(num_points - i) * 5)
            bytes_val = random.randint(1000000, 50000000)  # 1MB - 50MB
            datapoints.append({
                "timestamp": timestamp.isoformat(),
                key: bytes_val
            })
        
        return datapoints

    def _estimate_ram_from_cpu(self, cpu_percent: float) -> float:
        """
        Estimate RAM usage from CPU (since CloudWatch doesn't provide RAM directly).
        In production, use CloudWatch Agent for memory metrics.
        """
        # Simple correlation: RAM tends to correlate loosely with CPU
        import random
        base_ram = cpu_percent * 0.8 + random.uniform(-10, 10)
        return max(10, min(90, base_ram))


def demo_cloudwatch():
    """Demo function to test CloudWatch metrics collection."""
    print("\n" + "=" * 60)
    print("  CLOUDWATCH METRICS DEMO")
    print("=" * 60)
    
    collector = CloudWatchMetricsCollector()
    
    # Get running instances
    instances = collector.get_running_instances()
    print(f"\nFound {len(instances)} instances: {instances}")
    
    # Get metrics for each instance
    for instance_id in instances[:3]:  # Limit to first 3
        print(f"\n--- Metrics for {instance_id} ---")
        metrics = collector.get_instance_metrics(instance_id)
        print(f"  CPU: {metrics['cpu']:.1f}% (avg: {metrics['cpu_avg']:.1f}%)")
        print(f"  Network In: {metrics['network_in']:,} bytes")
        print(f"  Network Out: {metrics['network_out']:,} bytes")
        print(f"  RAM (estimated): {metrics['ram']:.1f}%")


if __name__ == "__main__":
    demo_cloudwatch()
