"""
live_pipeline.py - AWS Production Pipeline
=============================================
Main orchestrator for running the ML-based VM migration on real AWS EC2.

Pipeline:
    CloudWatch Metrics → ML Prediction → Migration Decision → EC2 Scaling Action

Usage:
    # Run as continuous service
    python -m aws.live_pipeline --continuous
    
    # Run single iteration
    python -m aws.live_pipeline --once
    
    # Run in dry-run mode (default)
    python -m aws.live_pipeline --dry-run
"""

import os
import sys
import time
import datetime
import argparse
import signal
from typing import Dict, List, Optional
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get as cfg
from logger import setup_logger
from aws.cloudwatch_metrics import CloudWatchMetricsCollector
from aws.ec2_scaler import EC2Scaler
from aws.aws_config import AWSConfig

logger = setup_logger(__name__)

# Feature columns expected by the model
FEATURE_COLUMNS = ["cpu", "ram", "network", "total_host_cpu", "total_host_ram"]


class AWSLivePipeline:
    """
    Production pipeline that runs ML-based migration decisions on real AWS EC2.
    
    Flow:
      1. Fetch CloudWatch metrics for all managed instances
      2. Transform metrics into ML model features
      3. Predict overload using trained Random Forest model
      4. Execute scaling actions (scale-out/in)
      5. Log decisions and metrics
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the AWS Live Pipeline.
        
        Args:
            model_path: Path to trained ML model (default: model/trained_model.pkl)
            scaler_path: Path to feature scaler (default: model/scaler.pkl)
        """
        self.model_path = model_path or cfg("model.model_path", "model/trained_model.pkl")
        self.scaler_path = scaler_path or cfg("model.scaler_path", "model/scaler.pkl")
        
        # Load ML model and scaler
        self._load_model()
        
        # Initialize AWS components
        self.aws_config = AWSConfig()
        self.metrics_collector = CloudWatchMetricsCollector()
        self.scaler = EC2Scaler()
        
        # Pipeline settings
        self.poll_interval = cfg("aws.pipeline.poll_interval_seconds", 60)
        self.prediction_threshold = cfg("aws.pipeline.prediction_threshold", 0.7)
        
        # State tracking
        self.is_running = False
        self.iteration_count = 0
        self.metrics_history = []
        self.decision_log = []
        
        logger.info("[Pipeline] AWS Live Pipeline initialized")

    def _load_model(self):
        """Load the trained ML model and scaler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Run training first: python main.py --step 3"
            )
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at '{self.scaler_path}'. "
                "Run preprocessing first: python main.py --step 2"
            )
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"[Pipeline] Loaded model from {self.model_path}")

    # ------------------------------------------------------------------ #
    #  Core Pipeline Steps
    # ------------------------------------------------------------------ #
    def fetch_metrics(self) -> List[Dict]:
        """
        Step 1: Fetch current CloudWatch metrics for all instances.
        
        Returns:
            List of metric dictionaries per instance
        """
        logger.info("[Pipeline] Step 1: Fetching CloudWatch metrics...")
        
        # Get running instances
        instance_ids = self.metrics_collector.get_running_instances()
        if not instance_ids:
            logger.warning("[Pipeline] No running instances found")
            return []
        
        logger.info(f"[Pipeline] Found {len(instance_ids)} instances: {instance_ids[:5]}...")
        
        # Fetch metrics for each instance
        metrics = self.metrics_collector.get_all_instances_metrics(instance_ids)
        
        # Store in history
        self.metrics_history.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metrics": metrics
        })
        # Keep only last 100 iterations
        self.metrics_history = self.metrics_history[-100:]
        
        return metrics

    def transform_features(self, instance_metrics: Dict) -> np.ndarray:
        """
        Step 2: Transform CloudWatch metrics into ML model features.
        
        Args:
            instance_metrics: Metrics dict from CloudWatchMetricsCollector
            
        Returns:
            Numpy array of scaled features
        """
        # Map CloudWatch metrics to model features
        features = {
            "cpu": instance_metrics.get("cpu", 0),
            "ram": instance_metrics.get("ram", 50),  # Estimated
            "network": instance_metrics.get("network_total", 0) / 1e6,  # Convert to MB
            "total_host_cpu": instance_metrics.get("cpu", 0),  # Same as instance CPU
            "total_host_ram": instance_metrics.get("ram", 50),  # Same as instance RAM
        }
        
        # Create feature vector
        values = np.array([[features[col] for col in FEATURE_COLUMNS]])
        
        # Scale using the trained scaler
        values_scaled = self.scaler.transform(values)
        
        return values_scaled

    def predict_overload(self, instance_metrics: Dict) -> Dict:
        """
        Step 3: Predict if instance is overloaded using ML model.
        
        Args:
            instance_metrics: Metrics dict for a single instance
            
        Returns:
            Dict with prediction, probability, and confidence
        """
        features_scaled = self.transform_features(instance_metrics)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        overload_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        confidence = max(probabilities)
        
        is_overloaded = (prediction == 1) or (overload_prob >= self.prediction_threshold)
        
        return {
            "instance_id": instance_metrics.get("instance_id"),
            "prediction": int(prediction),
            "is_overloaded": is_overloaded,
            "overload_probability": round(overload_prob, 4),
            "confidence": round(confidence, 4)
        }

    def make_decision(self, predictions: List[Dict], metrics: List[Dict]) -> Dict:
        """
        Step 4: Decide on scaling actions based on predictions.
        
        Args:
            predictions: List of prediction results per instance
            metrics: List of metrics per instance
            
        Returns:
            Dict with decision and recommended actions
        """
        logger.info("[Pipeline] Step 4: Making scaling decision...")
        
        overloaded = [p for p in predictions if p["is_overloaded"]]
        healthy = [p for p in predictions if not p["is_overloaded"]]
        
        decision = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "total_instances": len(predictions),
            "overloaded_count": len(overloaded),
            "healthy_count": len(healthy),
            "action": "none",
            "action_details": {}
        }
        
        # Determine action
        if len(overloaded) > 0:
            decision["action"] = "scale_out"
            decision["action_details"] = {
                "reason": f"{len(overloaded)} instances predicted overloaded",
                "instances": [p["instance_id"] for p in overloaded],
                "recommended_new_instances": min(len(overloaded), 2)
            }
        elif len(healthy) > 2 and self._all_underutilized(metrics, threshold=20):
            decision["action"] = "scale_in"
            decision["action_details"] = {
                "reason": "All instances underutilized",
                "candidates": [m["instance_id"] for m in metrics if m["cpu"] < 20][:1]
            }
        else:
            decision["action"] = "none"
            decision["action_details"] = {"reason": "System stable"}
        
        logger.info(f"[Pipeline] Decision: {decision['action']} - {decision['action_details'].get('reason')}")
        
        return decision

    def execute_action(self, decision: Dict, metrics: List[Dict], predictions: List[Dict]) -> Dict:
        """
        Step 5: Execute the scaling action.
        
        Args:
            decision: Decision dict from make_decision()
            metrics: Current metrics
            predictions: Current predictions
            
        Returns:
            Dict with execution result
        """
        logger.info(f"[Pipeline] Step 5: Executing action: {decision['action']}")
        
        result = {
            "action": decision["action"],
            "success": False,
            "details": {}
        }
        
        if decision["action"] == "scale_out":
            count = decision["action_details"].get("recommended_new_instances", 1)
            launched = self.scaler.scale_out(
                count=count,
                reason=decision["action_details"].get("reason", "ML prediction")
            )
            result["success"] = len(launched) > 0
            result["details"]["launched_instances"] = launched
            
        elif decision["action"] == "scale_in":
            candidates = decision["action_details"].get("candidates", [])
            if candidates:
                terminated = self.scaler.scale_in(
                    instance_ids=candidates,
                    reason=decision["action_details"].get("reason", "Underutilized")
                )
                result["success"] = len(terminated) > 0
                result["details"]["terminated_instances"] = terminated
        else:
            result["success"] = True
            result["details"]["message"] = "No action needed"
        
        # Log decision
        self.decision_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "decision": decision,
            "result": result
        })
        # Keep only last 100 decisions
        self.decision_log = self.decision_log[-100:]
        
        return result

    def _all_underutilized(self, metrics: List[Dict], threshold: float = 20) -> bool:
        """Check if all instances are underutilized."""
        if not metrics:
            return False
        return all(m.get("cpu", 100) < threshold for m in metrics)

    # ------------------------------------------------------------------ #
    #  Pipeline Execution
    # ------------------------------------------------------------------ #
    def run_once(self) -> Dict:
        """
        Run a single iteration of the pipeline.
        
        Returns:
            Dict with full pipeline result
        """
        self.iteration_count += 1
        start_time = datetime.datetime.utcnow()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[Pipeline] ITERATION {self.iteration_count}")
        logger.info(f"{'='*60}")
        
        result = {
            "iteration": self.iteration_count,
            "timestamp": start_time.isoformat(),
            "metrics": [],
            "predictions": [],
            "decision": {},
            "execution": {},
            "duration_ms": 0
        }
        
        try:
            # Step 1: Fetch metrics
            metrics = self.fetch_metrics()
            result["metrics"] = metrics
            
            if not metrics:
                logger.warning("[Pipeline] No metrics available, skipping iteration")
                return result
            
            # Step 2-3: Transform and predict for each instance
            logger.info("[Pipeline] Step 2-3: Predicting overload...")
            predictions = []
            for m in metrics:
                pred = self.predict_overload(m)
                predictions.append(pred)
                
                status = "⚠️ OVERLOADED" if pred["is_overloaded"] else "✓ OK"
                logger.info(
                    f"  {pred['instance_id']}: CPU={m['cpu']:.1f}% "
                    f"→ P(overload)={pred['overload_probability']:.2%} {status}"
                )
            
            result["predictions"] = predictions
            
            # Step 4: Make decision
            decision = self.make_decision(predictions, metrics)
            result["decision"] = decision
            
            # Step 5: Execute action
            execution = self.execute_action(decision, metrics, predictions)
            result["execution"] = execution
            
        except Exception as e:
            logger.error(f"[Pipeline] Error in iteration: {e}")
            result["error"] = str(e)
        
        # Calculate duration
        duration = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000
        result["duration_ms"] = round(duration, 2)
        
        logger.info(f"[Pipeline] Iteration complete in {result['duration_ms']}ms")
        
        return result

    def run_continuous(self):
        """
        Run the pipeline continuously at specified interval.
        
        Press Ctrl+C to stop.
        """
        logger.info(f"[Pipeline] Starting continuous mode (interval: {self.poll_interval}s)")
        logger.info("[Pipeline] Press Ctrl+C to stop")
        
        self.is_running = True
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\n[Pipeline] Received shutdown signal...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.is_running:
                result = self.run_once()
                
                if not self.is_running:
                    break
                
                logger.info(f"[Pipeline] Sleeping for {self.poll_interval}s...")
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("\n[Pipeline] Interrupted by user")
        finally:
            logger.info("[Pipeline] Pipeline stopped")
            self._cleanup()

    def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("[Pipeline] Cleaning up...")
        # Don't automatically terminate instances on shutdown
        # Just log the state
        managed = self.scaler.get_managed_instances()
        if managed:
            logger.info(f"[Pipeline] Managed instances still running: {managed}")
            logger.info("[Pipeline] Run cleanup manually if needed: scaler.cleanup_all()")

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            "is_running": self.is_running,
            "iteration_count": self.iteration_count,
            "managed_instances": self.scaler.get_managed_instances(),
            "last_metrics": self.metrics_history[-1] if self.metrics_history else None,
            "last_decision": self.decision_log[-1] if self.decision_log else None,
            "aws_config": {
                "region": self.aws_config.region,
                "dry_run": self.aws_config.dry_run
            }
        }


def main():
    """Main entry point for AWS Live Pipeline."""
    parser = argparse.ArgumentParser(
        description="AWS Live Pipeline - ML-based VM Migration"
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously at configured interval"
    )
    parser.add_argument(
        "--once", "-o",
        action="store_true",
        help="Run single iteration"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Force dry-run mode (no actual AWS changes)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=None,
        help="Poll interval in seconds (overrides config)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show AWS configuration and exit"
    )
    
    args = parser.parse_args()
    
    # If --status, just show config
    if args.status:
        aws_cfg = AWSConfig()
        aws_cfg.print_configuration()
        return
    
    print("\n" + "=" * 60)
    print("  AWS LIVE PIPELINE - ML-Based VM Migration")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = AWSLivePipeline()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease train the model first:")
        print("  python main.py")
        print("  Select option [3] to train the model")
        return
    
    # Override interval if specified
    if args.interval:
        pipeline.poll_interval = args.interval
    
    # Run pipeline
    if args.continuous:
        pipeline.run_continuous()
    else:
        # Default: single iteration
        result = pipeline.run_once()
        
        print("\n" + "-" * 60)
        print("  PIPELINE RESULT")
        print("-" * 60)
        print(f"  Instances checked:  {len(result['metrics'])}")
        print(f"  Overloaded:         {result['decision'].get('overloaded_count', 0)}")
        print(f"  Action taken:       {result['decision'].get('action', 'none')}")
        print(f"  Duration:           {result['duration_ms']}ms")
        
        if result['execution'].get('details'):
            print(f"  Details:            {result['execution']['details']}")


if __name__ == "__main__":
    main()
