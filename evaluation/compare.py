"""
compare.py - Strategy Comparison
===================================
Compares Rule-Based vs ML-Based migration strategies side by side.
Runs both strategies on the same simulation data and plots comparison charts.
"""

import os
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulation.simulator import Simulator
from simulation.migration import migrate_vm, clear_migration_log, get_migration_log
from decision.engine import DecisionEngine
from evaluation.evaluate import evaluate_strategy


def run_rule_based_strategy(simulator, num_ticks=10, overload_threshold=85):
    """
    Run a simple rule-based migration strategy.
    
    Rule: If host CPU > threshold, migrate the VM with highest CPU
    to the least-loaded host.
    
    Args:
        simulator (Simulator): Simulator instance.
        num_ticks (int): Number of ticks to run.
        overload_threshold (float): CPU threshold for overload.
        
    Returns:
        int: Number of migrations performed.
    """
    print(f"\n{'=' * 60}")
    print(f"RULE-BASED STRATEGY (Threshold: {overload_threshold}%)")
    print(f"{'=' * 60}")

    migration_count = 0
    clear_migration_log()

    for tick in range(1, num_ticks + 1):
        # Update VM usage
        for host in simulator.hosts:
            for vm in host.vms:
                vm.update_usage()

        # Check each host
        for host in simulator.hosts:
            if host.get_total_cpu() > overload_threshold and len(host.vms) > 1:
                # Select VM with highest CPU
                vm = max(host.vms, key=lambda v: v.cpu)
                target = simulator.get_least_loaded_host(exclude_host_id=host.host_id)
                if target:
                    migrate_vm(host, target, vm.vm_id)
                    migration_count += 1

    return migration_count


def run_ml_based_strategy(simulator, num_ticks=10):
    """
    Run the ML-based migration strategy using the Decision Engine.
    
    Args:
        simulator (Simulator): Simulator instance.
        num_ticks (int): Number of ticks to run.
        
    Returns:
        int: Number of migrations performed.
    """
    clear_migration_log()
    engine = DecisionEngine()
    migration_count = engine.run(simulator, num_ticks=num_ticks)
    return migration_count


def compare_strategies(num_ticks=15):
    """
    Compare Rule-Based vs ML-Based migration strategies.
    
    Creates two separate simulators with the same random seed,
    runs each strategy, then evaluates and plots comparison.
    
    Args:
        num_ticks (int): Number of ticks for each strategy.
        
    Returns:
        tuple: (rule_results, ml_results)
    """
    import random
    seed = 42

    # ---- Run Rule-Based ----
    random.seed(seed)
    np.random.seed(seed)
    sim_rule = Simulator(num_hosts=5, num_vms=20)
    rule_migrations = run_rule_based_strategy(sim_rule, num_ticks=num_ticks)
    rule_results = evaluate_strategy(sim_rule.hosts, rule_migrations, "Rule-Based")

    # ---- Run ML-Based ----
    random.seed(seed)
    np.random.seed(seed)
    sim_ml = Simulator(num_hosts=5, num_vms=20)
    ml_migrations = run_ml_based_strategy(sim_ml, num_ticks=num_ticks)
    ml_results = evaluate_strategy(sim_ml.hosts, ml_migrations, "ML-Based")

    # ---- Plot Comparison ----
    plot_comparison(rule_results, ml_results)

    return rule_results, ml_results


def plot_comparison(rule_results, ml_results):
    """
    Plot side-by-side bar charts comparing both strategies.
    
    Args:
        rule_results (dict): Evaluation results from rule-based strategy.
        ml_results (dict): Evaluation results from ML-based strategy.
    """
    os.makedirs("evaluation", exist_ok=True)

    categories = ["SLA Violations", "Energy (W)", "Migrations"]
    rule_values = [
        rule_results["sla_violations"],
        rule_results["total_energy_watts"],
        rule_results["migration_count"],
    ]
    ml_values = [
        ml_results["sla_violations"],
        ml_results["total_energy_watts"],
        ml_results["migration_count"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Rule-Based vs ML-Based Migration Strategy Comparison",
                 fontsize=14, fontweight="bold")

    colors = [("#e74c3c", "#2ecc71"), ("#e67e22", "#3498db"), ("#9b59b6", "#1abc9c")]

    for i, (cat, rv, mv) in enumerate(zip(categories, rule_values, ml_values)):
        bars = axes[i].bar(
            ["Rule-Based", "ML-Based"],
            [rv, mv],
            color=[colors[i][0], colors[i][1]],
            edgecolor="white",
            linewidth=1.5,
            width=0.5
        )
        axes[i].set_title(cat, fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Value")

        # Add value labels on bars
        for bar, val in zip(bars, [rv, mv]):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    output_path = "evaluation/comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[COMPARISON] Chart saved to '{output_path}'")
