"""
engine.py - Decision Engine (Enhanced)
=========================================
Uses the trained ML model to predict host overload in real-time.

Supports two modes:
  - simulation: Uses simulated hosts/VMs (default)
  - aws:        Uses real EC2 instances with CloudWatch metrics

Supports two migration strategies:
  - simple:     Migrate highest-CPU VM (original behaviour)
  - cost_aware: Optimise VM selection to minimise a composite cost
                function of SLA violations, migration overhead and
                load imbalance.
"""

import os
import numpy as np
import joblib
from simulation.migration import migrate_vm, print_migration_summary
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)

FEATURE_COLUMNS = ["cpu", "ram", "network", "total_host_cpu", "total_host_ram"]

# Mode constants
MODE_SIMULATION = "simulation"
MODE_AWS = "aws"


class DecisionEngine:
    """
    ML-based decision engine for VM migration.

    Loads the trained model and scaler, then applies them to live
    host/VM metrics to predict overload and trigger migrations.
    
    Modes:
      - simulation: Works with simulated hosts/VMs (default)
      - aws: Works with real EC2 instances and CloudWatch metrics
    """

    def __init__(self, model_path=None, scaler_path=None, mode=None):
        model_path = model_path or cfg("model.model_path", "model/trained_model.pkl")
        scaler_path = scaler_path or cfg("model.scaler_path", "model/scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run training first."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at '{scaler_path}'. Run preprocessing first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Mode: simulation or aws
        self.mode = mode or cfg("decision.mode", MODE_SIMULATION)
        
        # Strategy config
        self.strategy = cfg("decision.strategy", "cost_aware")
        self.migration_cost = cfg("decision.migration_cost_penalty", 5.0)
        self.sla_weight = cfg("decision.sla_violation_weight", 10.0)
        self.balance_weight = cfg("decision.load_balance_weight", 2.0)
        
        # AWS components (lazy loaded)
        self._metrics_collector = None
        self._ec2_scaler = None

        logger.info(f"Decision engine loaded (mode={self.mode}, strategy={self.strategy}).")

    # ------------------------------------------------------------------ #
    #  AWS Mode Support
    # ------------------------------------------------------------------ #
    @property
    def metrics_collector(self):
        """Lazy-load CloudWatch metrics collector for AWS mode."""
        if self._metrics_collector is None:
            from aws.cloudwatch_metrics import CloudWatchMetricsCollector
            self._metrics_collector = CloudWatchMetricsCollector()
        return self._metrics_collector

    @property
    def ec2_scaler(self):
        """Lazy-load EC2 scaler for AWS mode."""
        if self._ec2_scaler is None:
            from aws.ec2_scaler import EC2Scaler
            self._ec2_scaler = EC2Scaler()
        return self._ec2_scaler

    def predict_overload_aws(self, instance_metrics):
        """
        Predict whether an EC2 instance is overloaded using CloudWatch metrics.
        
        Args:
            instance_metrics: Dict from CloudWatchMetricsCollector.get_instance_metrics()
            
        Returns:
            (is_overloaded, confidence, probability)
        """
        features = {
            "cpu": instance_metrics.get("cpu", 0),
            "ram": instance_metrics.get("ram", 50),
            "network": instance_metrics.get("network_total", 0) / 1e6,  # MB
            "total_host_cpu": instance_metrics.get("cpu", 0),
            "total_host_ram": instance_metrics.get("ram", 50),
        }
        values = np.array([[features[col] for col in FEATURE_COLUMNS]])
        values_scaled = self.scaler.transform(values)

        prediction = self.model.predict(values_scaled)[0]
        probabilities = self.model.predict_proba(values_scaled)[0]
        confidence = max(probabilities)
        overload_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]

        return bool(prediction == 1), confidence, overload_prob

    def run_aws(self, num_iterations=10, interval_seconds=60):
        """
        Run the decision engine in AWS mode.
        
        Args:
            num_iterations: Number of iterations to run (0 for continuous)
            interval_seconds: Seconds between iterations
            
        Returns:
            Dict with scaling actions summary
        """
        import time
        
        logger.info(f"DECISION ENGINE [AWS MODE] -- Running for {num_iterations} iterations")
        
        total_scale_out = 0
        total_scale_in = 0
        
        iteration = 0
        while num_iterations == 0 or iteration < num_iterations:
            iteration += 1
            logger.info(f"\n--- AWS Iteration {iteration} ---")
            
            # Get all running instances
            instance_ids = self.metrics_collector.get_running_instances()
            if not instance_ids:
                logger.warning("No running instances found")
                if num_iterations > 0:
                    time.sleep(interval_seconds)
                continue
            
            # Fetch metrics and make predictions
            predictions = {}
            all_metrics = []
            
            for instance_id in instance_ids:
                metrics = self.metrics_collector.get_instance_metrics(instance_id)
                all_metrics.append(metrics)
                
                is_overloaded, confidence, prob = self.predict_overload_aws(metrics)
                predictions[instance_id] = is_overloaded
                
                status = "⚠️ OVERLOADED" if is_overloaded else "✓ OK"
                logger.info(
                    f"  {instance_id}: CPU={metrics['cpu']:.1f}% "
                    f"P(overload)={prob:.2%} {status}"
                )
            
            # Execute scaling actions
            actions = self.ec2_scaler.evaluate_and_scale(all_metrics, predictions)
            
            total_scale_out += len(actions.get("scale_out", []))
            total_scale_in += len(actions.get("scale_in", []))
            
            if actions["scale_out"]:
                logger.info(f"  SCALED OUT: {actions['scale_out']}")
            if actions["scale_in"]:
                logger.info(f"  SCALED IN: {actions['scale_in']}")
            
            if num_iterations > 0 and iteration < num_iterations:
                time.sleep(interval_seconds)
        
        logger.info(f"\nAWS MODE COMPLETE -- Scale-out: {total_scale_out}, Scale-in: {total_scale_in}")
        
        return {
            "total_scale_out": total_scale_out,
            "total_scale_in": total_scale_in,
            "managed_instances": self.ec2_scaler.get_managed_instances()
        }

    def predict_overload(self, host):
        """Predict whether a host is overloaded using the ML model."""
        features = {
            "cpu": host.get_total_cpu(),
            "ram": host.get_total_ram(),
            "network": host.get_total_network(),
            "total_host_cpu": host.get_total_cpu(),
            "total_host_ram": host.get_total_ram(),
        }
        values = np.array([[features[col] for col in FEATURE_COLUMNS]])
        values_scaled = self.scaler.transform(values)

        prediction = self.model.predict(values_scaled)[0]
        probabilities = self.model.predict_proba(values_scaled)[0]
        confidence = max(probabilities)

        return bool(prediction == 1), confidence

    # ------------------------------------------------------------------ #
    #  VM selection strategies
    # ------------------------------------------------------------------ #
    def select_vm_to_migrate(self, host, simulator=None):
        """
        Select the VM to migrate based on current strategy.

        Args:
            host (Host): Overloaded host.
            simulator (Simulator): Full simulator (needed for cost_aware).

        Returns:
            VM or None
        """
        if not host.vms:
            return None

        if self.strategy == "cost_aware" and simulator is not None:
            return self._select_cost_aware(host, simulator)
        else:
            # Fallback: highest-CPU VM (original behaviour)
            return max(host.vms, key=lambda vm: vm.cpu)

    def _select_cost_aware(self, host, simulator):
        """
        Cost-aware VM selection: evaluate the cost of migrating each VM
        and pick the one that minimises a composite cost function.

        Cost = sla_weight * SLA_violations_after
             + balance_weight * load_variance_after
             + migration_cost_penalty
        """
        best_vm = None
        best_cost = float("inf")

        target = simulator.get_least_loaded_host(exclude_host_id=host.host_id)
        if target is None:
            return max(host.vms, key=lambda vm: vm.cpu)

        sla_threshold = cfg("evaluation.sla_threshold", 90)

        for vm in host.vms:
            # Simulate the migration effect
            src_cpu_after = self._cpu_without_vm(host, vm)
            tgt_cpu_after = self._cpu_with_vm(target, vm)

            # SLA violations
            sla_violations = 0
            for h in simulator.hosts:
                cpu = h.get_total_cpu()
                if h.host_id == host.host_id:
                    cpu = src_cpu_after
                elif h.host_id == target.host_id:
                    cpu = tgt_cpu_after
                if cpu > sla_threshold:
                    sla_violations += 1

            # Load imbalance (variance of CPU across all hosts)
            cpus = []
            for h in simulator.hosts:
                if h.host_id == host.host_id:
                    cpus.append(src_cpu_after)
                elif h.host_id == target.host_id:
                    cpus.append(tgt_cpu_after)
                else:
                    cpus.append(h.get_total_cpu())
            load_variance = np.var(cpus)

            # Composite cost
            cost = (self.sla_weight * sla_violations
                    + self.balance_weight * load_variance
                    + self.migration_cost)

            if cost < best_cost:
                best_cost = cost
                best_vm = vm

        return best_vm

    @staticmethod
    def _cpu_without_vm(host, vm):
        """Estimate host CPU if a VM were removed."""
        remaining = [v.cpu for v in host.vms if v.vm_id != vm.vm_id]
        return sum(remaining) / len(remaining) if remaining else 0.0

    @staticmethod
    def _cpu_with_vm(host, vm):
        """Estimate host CPU if a VM were added."""
        all_cpus = [v.cpu for v in host.vms] + [vm.cpu]
        return sum(all_cpus) / len(all_cpus)

    # ------------------------------------------------------------------ #
    #  Main run loop
    # ------------------------------------------------------------------ #
    def run(self, simulator, num_ticks=10):
        """
        Run the decision engine over multiple simulation ticks.
        """
        logger.info(f"DECISION ENGINE -- Running for {num_ticks} ticks (strategy={self.strategy})")

        total_migrations = 0

        for tick in range(1, num_ticks + 1):
            # Update VM usage
            for host in simulator.hosts:
                for vm in host.vms:
                    vm.update_usage()

            # Check each host for overload
            for host in simulator.hosts:
                is_overloaded, confidence = self.predict_overload(host)

                if is_overloaded and len(host.vms) > 1:
                    logger.info(
                        f"  Tick {tick} | {host.host_id} OVERLOADED "
                        f"(CPU: {host.get_total_cpu():.1f}%, conf: {confidence:.2f})"
                    )

                    vm = self.select_vm_to_migrate(host, simulator)
                    if vm:
                        target = simulator.get_least_loaded_host(
                            exclude_host_id=host.host_id
                        )
                        if target:
                            migrate_vm(host, target, vm.vm_id)
                            total_migrations += 1

        logger.info(f"DECISION ENGINE COMPLETE -- Total migrations: {total_migrations}")
        print_migration_summary()

        return total_migrations
