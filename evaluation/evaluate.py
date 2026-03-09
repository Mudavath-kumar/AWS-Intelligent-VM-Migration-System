"""
evaluate.py - Evaluation Module (Enhanced)
=============================================
Calculates comprehensive metrics for evaluating VM migration strategies:
- SLA violations
- Dynamic power model energy consumption
- Migration overhead cost
- Load imbalance variance
- Energy efficiency score
- PR-AUC, ROC-AUC (via model evaluation)
"""

import numpy as np
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)


def calculate_sla_violations(hosts, sla_threshold=None):
    """
    Count hosts exceeding the SLA CPU threshold.

    Returns:
        tuple: (violation_count, violation_details)
    """
    sla_threshold = sla_threshold or cfg("evaluation.sla_threshold", 90)
    violations = []
    for host in hosts:
        cpu = host.get_total_cpu()
        if cpu > sla_threshold:
            violations.append({
                "host_id": host.host_id,
                "cpu": round(cpu, 2),
                "num_vms": len(host.vms),
            })
    return len(violations), violations


def calculate_energy_consumption(hosts):
    """
    Dynamic power model:
        Power = P_idle + (P_max - P_idle) * utilization

    Uses host-level power model (per-host P_idle / P_max).
    Hosts with no VMs consume 30% of idle power.

    Returns:
        tuple: (total_energy, per_host_energy)
    """
    per_host = {}
    total = 0.0

    for host in hosts:
        energy = host.get_power_consumption()
        per_host[host.host_id] = round(energy, 2)
        total += energy

    return round(total, 2), per_host


def calculate_migration_overhead(migration_count, cost_per_migration=None):
    """
    Calculate total migration overhead cost.

    Args:
        migration_count (int): Number of migrations.
        cost_per_migration (float): Cost per migration event.

    Returns:
        float: Total migration overhead.
    """
    cost_per_migration = cost_per_migration or cfg("decision.migration_cost_penalty", 5.0)
    return round(migration_count * cost_per_migration, 2)


def calculate_load_imbalance(hosts):
    """
    Calculate load imbalance as variance of CPU utilization across hosts.
    Lower variance = better load balance.

    Returns:
        float: Variance of host CPU utilizations.
    """
    cpus = [host.get_total_cpu() for host in hosts]
    return round(float(np.var(cpus)), 4)


def calculate_energy_efficiency(hosts, migration_count):
    """
    Energy efficiency score = Total useful work / (Energy + Migration overhead).
    Higher is better.

    Returns:
        float: Energy efficiency score.
    """
    total_work = sum(host.get_total_cpu() * len(host.vms) for host in hosts)
    total_energy, _ = calculate_energy_consumption(hosts)
    migration_overhead = calculate_migration_overhead(migration_count)

    denominator = total_energy + migration_overhead
    if denominator == 0:
        return 0.0
    return round(total_work / denominator, 4)


def evaluate_strategy(hosts, migration_count, strategy_name="Strategy"):
    """
    Full evaluation of a migration strategy with advanced metrics.

    Returns:
        dict: Comprehensive evaluation results.
    """
    sla_violations, violation_details = calculate_sla_violations(hosts)
    total_energy, per_host_energy = calculate_energy_consumption(hosts)
    migration_overhead = calculate_migration_overhead(migration_count)
    load_imbalance = calculate_load_imbalance(hosts)
    energy_efficiency = calculate_energy_efficiency(hosts, migration_count)

    results = {
        "strategy": strategy_name,
        "sla_violations": sla_violations,
        "total_energy_watts": total_energy,
        "migration_count": migration_count,
        "migration_overhead": migration_overhead,
        "load_imbalance_variance": load_imbalance,
        "energy_efficiency_score": energy_efficiency,
        "per_host_energy": per_host_energy,
        "violation_details": violation_details,
    }

    # Print results
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  EVALUATION -- {strategy_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  SLA Violations:          {sla_violations}")
    logger.info(f"  Total Energy:            {total_energy} W")
    logger.info(f"  Total Migrations:        {migration_count}")
    logger.info(f"  Migration Overhead:      {migration_overhead}")
    logger.info(f"  Load Imbalance (var):    {load_imbalance}")
    logger.info(f"  Energy Efficiency:       {energy_efficiency}")
    logger.info(f"{'-' * 60}")
    logger.info(f"  Per-Host Energy:")
    for host_id, energy in per_host_energy.items():
        logger.info(f"    {host_id}: {energy} W")
    if violation_details:
        logger.info(f"  SLA Violation Details:")
        for v in violation_details:
            logger.info(f"    {v['host_id']}: CPU={v['cpu']}%, VMs={v['num_vms']}")
    logger.info(f"{'=' * 60}")

    return results
