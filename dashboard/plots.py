"""
plots.py - Visualization Dashboard
=====================================
Generates publication-quality plots for the VM migration project:
- CPU/RAM usage trends over time
- Migration frequency histogram
- Energy consumption comparison
- Host utilization heatmap
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set global style
sns.set_theme(style="whitegrid", palette="muted")


def plot_cpu_ram_trends(csv_path="data/vm_metrics.csv", output_dir="dashboard"):
    """
    Plot CPU and RAM usage trends over simulation ticks.
    Shows average CPU/RAM per host across time.
    
    Args:
        csv_path (str): Path to metrics CSV.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Aggregate by tick and host
    host_agg = df.groupby(["tick", "host_id"]).agg(
        avg_cpu=("cpu", "mean"),
        avg_ram=("ram", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CPU & RAM Usage Trends Over Time", fontsize=14, fontweight="bold")

    # CPU Trends
    for host_id in sorted(host_agg["host_id"].unique()):
        data = host_agg[host_agg["host_id"] == host_id]
        ax1.plot(data["tick"], data["avg_cpu"], label=host_id, linewidth=2, alpha=0.8)

    ax1.axhline(y=85, color="red", linestyle="--", alpha=0.7, label="Overload Threshold (85%)")
    ax1.set_title("Average CPU Usage per Host", fontsize=12)
    ax1.set_xlabel("Simulation Tick")
    ax1.set_ylabel("CPU (%)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 100)

    # RAM Trends
    for host_id in sorted(host_agg["host_id"].unique()):
        data = host_agg[host_agg["host_id"] == host_id]
        ax2.plot(data["tick"], data["avg_ram"], label=host_id, linewidth=2, alpha=0.8)

    ax2.set_title("Average RAM Usage per Host", fontsize=12)
    ax2.set_xlabel("Simulation Tick")
    ax2.set_ylabel("RAM (%)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(output_dir, "cpu_ram_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] CPU/RAM trends saved to '{path}'")


def plot_migration_frequency(migration_log, output_dir="dashboard"):
    """
    Plot migration frequency — how many migrations per tick or time window.
    
    Args:
        migration_log (list[dict]): List of migration event dictionaries.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not migration_log:
        print("[DASHBOARD] No migration events to plot.")
        return

    # Count migrations per source host
    source_counts = {}
    target_counts = {}
    for event in migration_log:
        src = event["source_host"]
        tgt = event["target_host"]
        source_counts[src] = source_counts.get(src, 0) + 1
        target_counts[tgt] = target_counts.get(tgt, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VM Migration Frequency Analysis", fontsize=14, fontweight="bold")

    # Migrations FROM each host
    hosts = sorted(set(list(source_counts.keys()) + list(target_counts.keys())))
    from_vals = [source_counts.get(h, 0) for h in hosts]
    to_vals = [target_counts.get(h, 0) for h in hosts]

    bars1 = ax1.bar(hosts, from_vals, color="#e74c3c", edgecolor="white", width=0.5)
    ax1.set_title("Migrations FROM Each Host", fontsize=12)
    ax1.set_ylabel("Number of Migrations")
    ax1.set_xlabel("Host")
    for bar, val in zip(bars1, from_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha="center", fontweight="bold")

    # Migrations TO each host
    bars2 = ax2.bar(hosts, to_vals, color="#2ecc71", edgecolor="white", width=0.5)
    ax2.set_title("Migrations TO Each Host", fontsize=12)
    ax2.set_ylabel("Number of Migrations")
    ax2.set_xlabel("Host")
    for bar, val in zip(bars2, to_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "migration_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] Migration frequency saved to '{path}'")


def plot_energy_comparison(rule_energy, ml_energy, output_dir="dashboard"):
    """
    Plot energy consumption comparison between strategies.
    
    Args:
        rule_energy (dict): Per-host energy for rule-based strategy.
        ml_energy (dict): Per-host energy for ML-based strategy.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    hosts = sorted(set(list(rule_energy.keys()) + list(ml_energy.keys())))
    rule_vals = [rule_energy.get(h, 0) for h in hosts]
    ml_vals = [ml_energy.get(h, 0) for h in hosts]

    x = np.arange(len(hosts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, rule_vals, width, label="Rule-Based",
                   color="#e74c3c", edgecolor="white")
    bars2 = ax.bar(x + width / 2, ml_vals, width, label="ML-Based",
                   color="#2ecc71", edgecolor="white")

    ax.set_title("Energy Consumption: Rule-Based vs ML-Based",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Host")
    ax.set_ylabel("Energy (Watts)")
    ax.set_xticks(x)
    ax.set_xticklabels(hosts)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "energy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] Energy comparison saved to '{path}'")


def plot_host_utilization_heatmap(csv_path="data/vm_metrics.csv", output_dir="dashboard"):
    """
    Plot a heatmap of host CPU utilization over time.
    
    Args:
        csv_path (str): Path to metrics CSV.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Pivot: rows = hosts, columns = ticks, values = avg CPU
    pivot = df.groupby(["host_id", "tick"])["total_host_cpu"].mean().reset_index()
    heatmap_data = pivot.pivot(index="host_id", columns="tick", values="total_host_cpu")

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "CPU %"},
        linewidths=0.1,
        linecolor="white"
    )
    ax.set_title("Host CPU Utilization Heatmap Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Simulation Tick")
    ax.set_ylabel("Host")

    plt.tight_layout()
    path = os.path.join(output_dir, "host_utilization_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] Heatmap saved to '{path}'")


def generate_all_plots(csv_path="data/vm_metrics.csv", migration_log=None,
                       rule_energy=None, ml_energy=None):
    """
    Generate all dashboard plots at once.
    
    Args:
        csv_path (str): Path to metrics CSV.
        migration_log (list): Migration events.
        rule_energy (dict): Rule-based per-host energy.
        ml_energy (dict): ML-based per-host energy.
    """
    print(f"\n{'=' * 50}")
    print("GENERATING DASHBOARD PLOTS")
    print(f"{'=' * 50}")

    plot_cpu_ram_trends(csv_path)
    plot_host_utilization_heatmap(csv_path)

    if migration_log:
        plot_migration_frequency(migration_log)

    if rule_energy and ml_energy:
        plot_energy_comparison(rule_energy, ml_energy)

    print(f"{'=' * 50}")
    print("All plots saved to 'dashboard/' directory")
    print(f"{'=' * 50}")
