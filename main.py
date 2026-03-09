"""
main.py - Intelligent VM Migration Strategy Using Machine Learning
====================================================================
End-to-end pipeline that ties together all project modules.
Run this file to execute the complete workflow.

Supports two modes:
  - Simulation Mode: Test with simulated VMs and hosts
  - AWS Production Mode: Run on real EC2 with CloudWatch metrics

Usage:
    python main.py          # Run interactive menu
    python main.py --all    # Run all steps automatically
    python main.py --aws    # Run AWS production pipeline
"""

import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step1_simulate():
    """Step 1: Run VM/Host simulation and generate CSV dataset."""
    print("\n" + "#" * 60)
    print("  STEP 1: SIMULATION & DATA GENERATION")
    print("#" * 60)

    from simulation.simulator import Simulator

    sim = Simulator(num_hosts=5, num_vms=20)
    sim.print_status()
    records = sim.run_simulation(num_ticks=50, output_path="data/vm_metrics.csv")
    sim.print_status()
    return sim


def step2_preprocess():
    """Step 2: Preprocess the generated dataset."""
    print("\n" + "#" * 60)
    print("  STEP 2: DATA PREPROCESSING")
    print("#" * 60)

    from model.preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(
        csv_path="data/vm_metrics.csv"
    )
    print(f"\n  [OK] Data preprocessed successfully!")
    print(f"     Features: {features}")
    print(f"     Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def step3_train():
    """Step 3: Train the Random Forest ML model."""
    print("\n" + "#" * 60)
    print("  STEP 3: ML MODEL TRAINING")
    print("#" * 60)

    from model.train import train_model

    model, accuracy, features = train_model(
        csv_path="data/vm_metrics.csv",
        model_path="model/trained_model.pkl"
    )
    return model, accuracy


def step4_decision_engine():
    """Step 4: Run the ML-based decision engine with live simulation."""
    print("\n" + "#" * 60)
    print("  STEP 4: DECISION ENGINE (LIVE MIGRATION)")
    print("#" * 60)

    from simulation.simulator import Simulator
    from decision.engine import DecisionEngine

    sim = Simulator(num_hosts=5, num_vms=20)
    engine = DecisionEngine()

    print("\n--- Before Decision Engine ---")
    sim.print_status()

    total_migrations = engine.run(sim, num_ticks=10)

    print("\n--- After Decision Engine ---")
    sim.print_status()

    return sim, total_migrations


def step5_evaluate():
    """Step 5: Evaluate and compare strategies."""
    print("\n" + "#" * 60)
    print("  STEP 5: EVALUATION & COMPARISON")
    print("#" * 60)

    from evaluation.compare import compare_strategies

    rule_results, ml_results = compare_strategies(num_ticks=15)
    return rule_results, ml_results


def step6_visualize(rule_results=None, ml_results=None):
    """Step 6: Generate dashboard visualizations."""
    print("\n" + "#" * 60)
    print("  STEP 6: VISUALIZATION DASHBOARD")
    print("#" * 60)

    from dashboard.plots import generate_all_plots
    from simulation.migration import get_migration_log

    migration_log = get_migration_log()

    rule_energy = rule_results["per_host_energy"] if rule_results else None
    ml_energy = ml_results["per_host_energy"] if ml_results else None

    generate_all_plots(
        csv_path="data/vm_metrics.csv",
        migration_log=migration_log if migration_log else None,
        rule_energy=rule_energy,
        ml_energy=ml_energy,
    )


def step7_aws_demo():
    """Step 7: AWS EC2 Integration demo."""
    print("\n" + "#" * 60)
    print("  STEP 7: AWS EC2 INTEGRATION (OPTIONAL)")
    print("#" * 60)

    from aws.ec2_manager import demo_aws_integration
    demo_aws_integration()


def step8_aws_production():
    """Step 8: Run AWS Production Pipeline with real CloudWatch metrics."""
    print("\n" + "#" * 60)
    print("  STEP 8: AWS PRODUCTION PIPELINE")
    print("#" * 60)
    
    from aws.aws_config import AWSConfig
    from aws.live_pipeline import AWSLivePipeline
    
    # Show AWS configuration
    aws_cfg = AWSConfig()
    aws_cfg.print_configuration()
    
    # Check if production ready
    is_ready, issues = aws_cfg.is_production_ready()
    if not is_ready and not aws_cfg.dry_run:
        print("\n[WARNING] AWS not fully configured for production:")
        for key, issue in issues.items():
            print(f"  - {key}: {issue}")
        print("\nContinuing in dry-run mode...")
    
    try:
        pipeline = AWSLivePipeline()
        
        print("\n--- Running Single Pipeline Iteration ---")
        result = pipeline.run_once()
        
        print("\n--- Pipeline Result ---")
        print(f"  Instances checked:  {len(result['metrics'])}")
        print(f"  Overloaded:         {result['decision'].get('overloaded_count', 0)}")
        print(f"  Action taken:       {result['decision'].get('action', 'none')}")
        print(f"  Duration:           {result['duration_ms']}ms")
        
        return result
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease train the model first (Step 3)")
        return None


def step9_aws_continuous():
    """Step 9: Run AWS Pipeline in continuous mode."""
    print("\n" + "#" * 60)
    print("  STEP 9: AWS CONTINUOUS MONITORING")
    print("#" * 60)
    
    from aws.live_pipeline import AWSLivePipeline
    
    try:
        pipeline = AWSLivePipeline()
        
        print("\n[INFO] Starting continuous monitoring...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        pipeline.run_continuous()
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease train the model first (Step 3)")


def run_all():
    """Run all steps in sequence."""
    print("\n" + "=" * 60)
    print("  INTELLIGENT VM MIGRATION STRATEGY USING ML")
    print("  Running full pipeline...")
    print("=" * 60)

    # Step 1: Simulation
    sim = step1_simulate()

    # Step 2: Preprocessing
    step2_preprocess()

    # Step 3: Training
    model, accuracy = step3_train()

    # Step 4: Decision Engine
    sim, migrations = step4_decision_engine()

    # Step 5: Evaluation & Comparison
    rule_results, ml_results = step5_evaluate()

    # Step 6: Visualization
    step6_visualize(rule_results, ml_results)

    # Step 7: AWS Demo
    step7_aws_demo()

    print("\n" + "=" * 60)
    print("  [OK] ALL STEPS COMPLETE!")
    print("=" * 60)
    print("\nOutput files generated:")
    print("  [DATA] data/vm_metrics.csv          - Simulation dataset")
    print("  [MODEL] model/trained_model.pkl     - Trained ML model")
    print("  [MODEL] model/scaler.pkl            - Feature scaler")
    print("  [PLOT] model/confusion_matrix.png   - Model confusion matrix")
    print("  [PLOT] evaluation/comparison.png     - Strategy comparison chart")
    print("  [PLOT] dashboard/cpu_ram_trends.png  - CPU/RAM trends")
    print("  [PLOT] dashboard/host_utilization_heatmap.png - Heatmap")
    print("  [PLOT] dashboard/migration_frequency.png - Migration frequency")
    print("  [PLOT] dashboard/energy_comparison.png - Energy comparison")


def show_menu():
    """Display interactive menu."""
    print("\n" + "=" * 60)
    print("  INTELLIGENT VM MIGRATION STRATEGY USING ML")
    print("=" * 60)
    print()
    print("  === SIMULATION MODE ===")
    print("  [1] Run Simulation & Generate CSV Data")
    print("  [2] Preprocess Dataset")
    print("  [3] Train ML Model (Random Forest)")
    print("  [4] Run Decision Engine (Live Migration)")
    print("  [5] Evaluate & Compare Strategies")
    print("  [6] Generate Visualization Plots")
    print("  [7] AWS EC2 Integration (Demo)")
    print()
    print("  === AWS PRODUCTION MODE ===")
    print("  [8] Run AWS Production Pipeline (Single)")
    print("  [9] Run AWS Continuous Monitoring")
    print()
    print("  === BATCH ===")
    print("  [A] Run ALL Simulation Steps (1-7)")
    print("  [0] Exit")
    print()
    return input("  Select an option (0-9, A): ").strip()


def main():
    """Main entry point with interactive menu or auto-run."""
    # Check for command line flags
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--all":
            run_all()
            return
        elif arg == "--aws":
            step8_aws_production()
            return
        elif arg == "--aws-continuous":
            step9_aws_continuous()
            return
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            return

    rule_results = None
    ml_results = None

    while True:
        choice = show_menu().upper()

        if choice == "1":
            step1_simulate()
        elif choice == "2":
            step2_preprocess()
        elif choice == "3":
            step3_train()
        elif choice == "4":
            step4_decision_engine()
        elif choice == "5":
            rule_results, ml_results = step5_evaluate()
        elif choice == "6":
            step6_visualize(rule_results, ml_results)
        elif choice == "7":
            step7_aws_demo()
        elif choice == "8":
            step8_aws_production()
        elif choice == "9":
            step9_aws_continuous()
        elif choice == "A":
            run_all()
        elif choice == "0":
            print("\n  Goodbye!")
            break
        else:
            print("  [ERROR] Invalid option. Please try again.")

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
