"""
simulator.py - Main Simulation Orchestrator (Enhanced)
========================================================
Creates heterogeneous physical hosts and virtual machines.
Runs simulation ticks with configurable workload patterns.
Collects all metrics into records for CSV export.
"""

import random
import numpy as np
from simulation.vm import VM
from simulation.host import Host
from simulation.data_generator import save_metrics_to_csv
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)


class Simulator:
    """
    Orchestrates the VM-Host simulation environment.

    Supports heterogeneous hosts (different CPU/RAM capacities)
    and configurable workload patterns per VM.
    """

    def __init__(self, num_hosts=None, num_vms=None):
        self.num_hosts = num_hosts or cfg("simulation.num_hosts", 5)
        self.num_vms = num_vms or cfg("simulation.num_vms", 20)
        self.hosts = []
        self.all_vms = []
        self.records = []
        self._setup()

    def _setup(self):
        """Create heterogeneous hosts and distribute VMs evenly."""
        host_caps = cfg("simulation.host_capacities", None)

        if host_caps and len(host_caps) >= self.num_hosts:
            # Use configured heterogeneous capacities
            for i, hc in enumerate(host_caps[:self.num_hosts]):
                host = Host(
                    host_id=hc.get("host_id", f"Host_{i+1}"),
                    cpu_capacity=hc.get("cpu_capacity", 100),
                    ram_capacity=hc.get("ram_capacity", 100),
                    overload_threshold=hc.get("overload_threshold", 85),
                )
                self.hosts.append(host)
        else:
            # Fallback: uniform hosts
            for i in range(1, self.num_hosts + 1):
                host = Host(host_id=f"Host_{i}")
                self.hosts.append(host)

        # Create VMs and distribute round-robin
        for i in range(1, self.num_vms + 1):
            vm = VM(vm_id=f"VM_{i:02d}")
            host_index = (i - 1) % self.num_hosts
            self.hosts[host_index].add_vm(vm)
            self.all_vms.append(vm)

        logger.info(f"Initialized {self.num_hosts} hosts with {self.num_vms} VMs")
        for host in self.hosts:
            logger.info(f"  {host}")

    def run_tick(self, tick_number):
        """Run one simulation tick: update VMs, collect metrics."""
        tick_records = []

        for host in self.hosts:
            for vm in host.vms:
                vm.update_usage()

            host_cpu = host.get_total_cpu()
            host_ram = host.get_total_ram()
            overloaded = host.is_overloaded()

            for vm in host.vms:
                record = {
                    "tick": tick_number,
                    "host_id": host.host_id,
                    "vm_id": vm.vm_id,
                    "cpu": round(vm.cpu, 2),
                    "ram": round(vm.ram, 2),
                    "network": round(vm.network, 2),
                    "total_host_cpu": round(host_cpu, 2),
                    "total_host_ram": round(host_ram, 2),
                    "host_cpu_capacity": host.cpu_capacity,
                    "overloaded": int(overloaded),
                }
                tick_records.append(record)

        self.records.extend(tick_records)
        return tick_records

    def run_simulation(self, num_ticks=None, output_path=None):
        """Run full simulation and save to CSV."""
        num_ticks = num_ticks or cfg("simulation.num_ticks", 50)
        output_path = output_path or cfg("simulation.output_csv", "data/vm_metrics.csv")

        logger.info(f"Running simulation for {num_ticks} ticks...")
        self.records = []

        for tick in range(1, num_ticks + 1):
            self.run_tick(tick)

            if tick % 10 == 0:
                overloaded_hosts = sum(1 for h in self.hosts if h.is_overloaded())
                logger.info(f"  Tick {tick}/{num_ticks} -- Overloaded: {overloaded_hosts}/{self.num_hosts}")

        save_metrics_to_csv(self.records, output_path)
        logger.info(f"Simulation complete! {len(self.records)} records generated.")
        return self.records

    def print_status(self):
        """Print current status of all hosts and their VMs."""
        print("\n" + "=" * 70)
        print("CURRENT HOST STATUS")
        print("=" * 70)
        for host in self.hosts:
            status = "** OVERLOADED **" if host.is_overloaded() else "[Normal]"
            print(f"\n{host.host_id} (capacity={host.cpu_capacity}) {status}")
            print(f"  Total CPU: {host.get_total_cpu():.1f}% | "
                  f"Total RAM: {host.get_total_ram():.1f}% | "
                  f"VMs: {len(host.vms)} | "
                  f"Power: {host.get_power_consumption():.1f}W")
            for vm in host.vms:
                print(f"    {vm}")
        print("=" * 70)

    def get_host_by_id(self, host_id):
        """Find and return a host by its ID."""
        for host in self.hosts:
            if host.host_id == host_id:
                return host
        return None

    def get_least_loaded_host(self, exclude_host_id=None):
        """Find the host with the lowest CPU utilization."""
        candidates = [h for h in self.hosts if h.host_id != exclude_host_id]
        return min(candidates, key=lambda h: h.get_total_cpu())
