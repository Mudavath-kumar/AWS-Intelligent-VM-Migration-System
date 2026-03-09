"""
ec2_scaler.py - EC2 Auto Scaling Actions
==========================================
Executes scaling actions based on ML predictions:
  - Launch new EC2 instances (scale-out)
  - Terminate idle instances (scale-in)
  - Rebalance workload across instances

Usage:
    from aws.ec2_scaler import EC2Scaler
    scaler = EC2Scaler()
    scaler.scale_out(count=2)
    scaler.scale_in(instance_ids=["i-xxx"])
"""

import datetime
import time
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


class EC2Scaler:
    """
    Handles EC2 scaling actions triggered by ML-based migration decisions.
    
    Actions:
      - scale_out(): Launch new instances when overload predicted
      - scale_in(): Terminate idle instances to save costs
      - rebalance(): Redistribute workload (via Application Load Balancer)
    """

    def __init__(self, region: str = None):
        self.region = region or cfg("aws.region", "us-east-1")
        self.dry_run = cfg("aws.dry_run", True)
        
        # Configuration
        self.instance_type = cfg("aws.instance_type", "t2.micro")
        self.ami_id = cfg("aws.ami_id", "ami-0c02fb55956c7d316")
        self.max_instances = cfg("aws.max_instances", 10)
        self.min_instances = cfg("aws.min_instances", 2)
        self.key_name = cfg("aws.key_name", None)
        self.security_group_ids = cfg("aws.security_group_ids", [])
        self.subnet_id = cfg("aws.subnet_id", None)
        
        # Scaling thresholds
        self.scale_out_threshold = cfg("aws.scaling.scale_out_threshold", 80)
        self.scale_in_threshold = cfg("aws.scaling.scale_in_threshold", 20)
        self.cooldown_seconds = cfg("aws.scaling.cooldown_seconds", 300)
        self.idle_minutes_before_terminate = cfg("aws.scaling.idle_terminate_minutes", 30)
        
        # Track scaling events
        self.last_scale_out_time = None
        self.last_scale_in_time = None
        self.managed_instances = []
        
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not installed. Using dry-run mode.")
            self.ec2 = None
            return
            
        if self.dry_run:
            logger.info(f"[EC2Scaler] Dry-run mode enabled")
            self.ec2 = None
            return
            
        try:
            self.ec2 = boto3.client("ec2", region_name=self.region)
            logger.info(f"[EC2Scaler] Connected to region: {self.region}")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"[EC2Scaler] Credentials error: {e}")
            self.ec2 = None

    # ------------------------------------------------------------------ #
    #  Scale Out (Launch New Instances)
    # ------------------------------------------------------------------ #
    def scale_out(self, count: int = 1, reason: str = "ML-predicted overload") -> List[str]:
        """
        Launch new EC2 instances when overload is predicted.
        
        Args:
            count: Number of instances to launch
            reason: Reason for scaling (for logging)
            
        Returns:
            List of launched instance IDs
        """
        # Check cooldown
        if self.last_scale_out_time:
            elapsed = (datetime.datetime.utcnow() - self.last_scale_out_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.info(f"[EC2Scaler] Scale-out cooldown active ({self.cooldown_seconds - elapsed:.0f}s remaining)")
                return []
        
        # Check max instances
        current_count = len(self.managed_instances)
        if current_count + count > self.max_instances:
            count = max(0, self.max_instances - current_count)
            if count == 0:
                logger.warning(f"[EC2Scaler] Max instances ({self.max_instances}) reached")
                return []
        
        logger.info(f"[EC2Scaler] SCALE OUT: Launching {count} instance(s). Reason: {reason}")
        
        if self.dry_run or not self.ec2:
            return self._simulate_launch(count)
        
        return self._launch_instances(count)

    def _launch_instances(self, count: int) -> List[str]:
        """Actually launch EC2 instances."""
        try:
            launch_params = {
                "ImageId": self.ami_id,
                "InstanceType": self.instance_type,
                "MinCount": count,
                "MaxCount": count,
                "TagSpecifications": [{
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Project", "Value": "VM-Migration-ML"},
                        {"Key": "ManagedBy", "Value": "ML-Decision-Engine"},
                        {"Key": "LaunchTime", "Value": datetime.datetime.utcnow().isoformat()}
                    ]
                }]
            }
            
            # Optional parameters
            if self.key_name:
                launch_params["KeyName"] = self.key_name
            if self.security_group_ids:
                launch_params["SecurityGroupIds"] = self.security_group_ids
            if self.subnet_id:
                launch_params["SubnetId"] = self.subnet_id
            
            response = self.ec2.run_instances(**launch_params)
            
            instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
            self.managed_instances.extend(instance_ids)
            self.last_scale_out_time = datetime.datetime.utcnow()
            
            logger.info(f"[EC2Scaler] Launched instances: {instance_ids}")
            
            # Wait for instances to be running
            self._wait_for_instances(instance_ids, state="running")
            
            return instance_ids
            
        except ClientError as e:
            logger.error(f"[EC2Scaler] Failed to launch instances: {e}")
            return []

    def _simulate_launch(self, count: int) -> List[str]:
        """Simulate instance launch in dry-run mode."""
        instance_ids = [f"i-simulated-{datetime.datetime.utcnow().strftime('%H%M%S')}-{i}" 
                       for i in range(count)]
        self.managed_instances.extend(instance_ids)
        self.last_scale_out_time = datetime.datetime.utcnow()
        logger.info(f"[DRY-RUN] Simulated launch: {instance_ids}")
        return instance_ids

    # ------------------------------------------------------------------ #
    #  Scale In (Terminate Idle Instances)
    # ------------------------------------------------------------------ #
    def scale_in(self, instance_ids: List[str] = None, reason: str = "Idle/underutilized") -> List[str]:
        """
        Terminate idle EC2 instances to save costs.
        
        Args:
            instance_ids: Specific instances to terminate. If None, auto-select idle ones.
            reason: Reason for termination (for logging)
            
        Returns:
            List of terminated instance IDs
        """
        # Check cooldown
        if self.last_scale_in_time:
            elapsed = (datetime.datetime.utcnow() - self.last_scale_in_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.info(f"[EC2Scaler] Scale-in cooldown active ({self.cooldown_seconds - elapsed:.0f}s remaining)")
                return []
        
        # Check min instances
        current_count = len(self.managed_instances)
        if current_count <= self.min_instances:
            logger.info(f"[EC2Scaler] At minimum instances ({self.min_instances}), cannot scale in")
            return []
        
        if not instance_ids:
            logger.info("[EC2Scaler] No instances specified for scale-in")
            return []
        
        # Don't go below minimum
        max_to_terminate = current_count - self.min_instances
        instance_ids = instance_ids[:max_to_terminate]
        
        if not instance_ids:
            return []
        
        logger.info(f"[EC2Scaler] SCALE IN: Terminating {len(instance_ids)} instance(s). Reason: {reason}")
        
        if self.dry_run or not self.ec2:
            return self._simulate_terminate(instance_ids)
        
        return self._terminate_instances(instance_ids)

    def _terminate_instances(self, instance_ids: List[str]) -> List[str]:
        """Actually terminate EC2 instances."""
        try:
            self.ec2.terminate_instances(InstanceIds=instance_ids)
            
            for iid in instance_ids:
                if iid in self.managed_instances:
                    self.managed_instances.remove(iid)
            
            self.last_scale_in_time = datetime.datetime.utcnow()
            logger.info(f"[EC2Scaler] Terminated: {instance_ids}")
            return instance_ids
            
        except ClientError as e:
            logger.error(f"[EC2Scaler] Failed to terminate: {e}")
            return []

    def _simulate_terminate(self, instance_ids: List[str]) -> List[str]:
        """Simulate termination in dry-run mode."""
        for iid in instance_ids:
            if iid in self.managed_instances:
                self.managed_instances.remove(iid)
        self.last_scale_in_time = datetime.datetime.utcnow()
        logger.info(f"[DRY-RUN] Simulated termination: {instance_ids}")
        return instance_ids

    # ------------------------------------------------------------------ #
    #  Auto-Scaling Based on Metrics
    # ------------------------------------------------------------------ #
    def evaluate_and_scale(self, instance_metrics: List[Dict], predictions: Dict[str, bool]) -> Dict:
        """
        Evaluate current metrics and predictions, then execute scaling actions.
        
        Args:
            instance_metrics: List of metrics from CloudWatchMetricsCollector
            predictions: Dict mapping instance_id -> is_overloaded (from ML model)
            
        Returns:
            Dict with scaling actions taken
        """
        actions = {
            "scale_out": [],
            "scale_in": [],
            "total_instances": len(self.managed_instances)
        }
        
        if not instance_metrics:
            return actions
        
        # Count overloaded and idle instances
        overloaded_instances = []
        idle_instances = []
        
        for metrics in instance_metrics:
            instance_id = metrics.get("instance_id")
            cpu = metrics.get("cpu", 0)
            
            # Check ML prediction
            is_predicted_overload = predictions.get(instance_id, False)
            
            if is_predicted_overload or cpu > self.scale_out_threshold:
                overloaded_instances.append(instance_id)
            elif cpu < self.scale_in_threshold:
                idle_instances.append(instance_id)
        
        # Scale out if any overloaded
        if overloaded_instances:
            count = min(len(overloaded_instances), 2)  # Max 2 at a time
            launched = self.scale_out(count, reason=f"Overloaded instances: {overloaded_instances[:3]}")
            actions["scale_out"] = launched
        
        # Scale in if idle (only if no overload)
        elif idle_instances and len(idle_instances) > 1:
            # Keep at least one idle instance
            to_terminate = idle_instances[:-1]
            terminated = self.scale_in(to_terminate[:1], reason="Low utilization")  # Terminate 1 at a time
            actions["scale_in"] = terminated
        
        actions["total_instances"] = len(self.managed_instances)
        return actions

    # ------------------------------------------------------------------ #
    #  Rebalance Workload
    # ------------------------------------------------------------------ #
    def rebalance_workload(self, overloaded_instance: str, target_instances: List[str]) -> bool:
        """
        Rebalance workload from overloaded instance to others.
        
        In real production, this would involve:
          - Updating ALB/NLB target weights
          - Draining connections from overloaded instance
          - Shifting traffic to healthy instances
          
        For now, we log the intent and simulate the action.
        """
        logger.info(f"[EC2Scaler] REBALANCE: Shifting load from {overloaded_instance}")
        logger.info(f"[EC2Scaler] Target instances: {target_instances}")
        
        if self.dry_run or not self.ec2:
            logger.info("[DRY-RUN] Simulated workload rebalance")
            return True
        
        # In production, you would:
        # 1. Update ALB target group weights
        # 2. Use connection draining
        # 3. Route53 weighted routing changes
        # For now, just log the intent
        logger.info("[EC2Scaler] Workload rebalance logged (implement ALB integration for production)")
        return True

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _wait_for_instances(self, instance_ids: List[str], state: str = "running", 
                            timeout: int = 300):
        """Wait for instances to reach specified state."""
        if not self.ec2:
            return
            
        logger.info(f"[EC2Scaler] Waiting for {len(instance_ids)} instances to be {state}...")
        
        waiter = self.ec2.get_waiter(f"instance_{state}")
        try:
            waiter.wait(
                InstanceIds=instance_ids,
                WaiterConfig={"Delay": 15, "MaxAttempts": timeout // 15}
            )
            logger.info(f"[EC2Scaler] Instances are now {state}")
        except Exception as e:
            logger.warning(f"[EC2Scaler] Timeout waiting for instances: {e}")

    def get_managed_instances(self) -> List[str]:
        """Return list of instances managed by this scaler."""
        return self.managed_instances.copy()

    def cleanup_all(self):
        """Terminate all managed instances (cleanup)."""
        if self.managed_instances:
            logger.info(f"[EC2Scaler] Cleanup: Terminating all {len(self.managed_instances)} managed instances")
            self.scale_in(self.managed_instances.copy(), reason="Cleanup")
        else:
            logger.info("[EC2Scaler] No instances to cleanup")


def demo_scaler():
    """Demo function to test EC2 scaling."""
    print("\n" + "=" * 60)
    print("  EC2 SCALER DEMO")
    print("=" * 60)
    
    scaler = EC2Scaler()
    
    print(f"\nConfiguration:")
    print(f"  Instance type: {scaler.instance_type}")
    print(f"  Max instances: {scaler.max_instances}")
    print(f"  Min instances: {scaler.min_instances}")
    print(f"  Scale-out threshold: {scaler.scale_out_threshold}%")
    print(f"  Scale-in threshold: {scaler.scale_in_threshold}%")
    
    # Test scale out
    print("\n--- Testing Scale Out ---")
    launched = scaler.scale_out(count=2, reason="Demo test")
    print(f"Launched: {launched}")
    print(f"Managed instances: {scaler.get_managed_instances()}")
    
    # Simulate metrics and evaluate
    print("\n--- Testing Auto-Scale Evaluation ---")
    mock_metrics = [
        {"instance_id": launched[0] if launched else "i-test-1", "cpu": 85},
        {"instance_id": launched[1] if len(launched) > 1 else "i-test-2", "cpu": 25}
    ]
    mock_predictions = {m["instance_id"]: m["cpu"] > 80 for m in mock_metrics}
    
    actions = scaler.evaluate_and_scale(mock_metrics, mock_predictions)
    print(f"Actions taken: {actions}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    scaler.cleanup_all()


if __name__ == "__main__":
    demo_scaler()
