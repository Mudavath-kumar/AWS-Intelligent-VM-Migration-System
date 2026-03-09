"""
vm.py - Virtual Machine Simulation (Enhanced)
================================================
Represents a single Virtual Machine (VM) with dynamic resource usage.
Supports three workload generation modes:
  - random:      Original purely random fluctuation
  - time_series: Sinusoidal diurnal pattern + Gaussian noise
  - burst:       Sudden spikes with gradual cooldown

Each VM has CPU, RAM, and Network utilization that fluctuates over time.
"""

import math
import random
import numpy as np
from config import get as cfg


class VM:
    """
    Simulates a Virtual Machine with fluctuating resource usage.

    Attributes:
        vm_id (str): Unique identifier for the VM.
        cpu (float): Current CPU utilization (0-100%).
        ram (float): Current RAM utilization (0-100%).
        network (float): Current network bandwidth usage (0-100%).
        host_id (str): ID of the host this VM is assigned to.
        workload_pattern (str): "random", "time_series", or "burst".
        tick (int): Internal tick counter for time-series patterns.
        burst_remaining (int): Ticks remaining in current burst event.
        burst_cpu_extra (float): Extra CPU load during a burst.
    """

    def __init__(self, vm_id, host_id=None, cpu=None, ram=None, network=None):
        self.vm_id = vm_id
        self.host_id = host_id

        # Increased base ranges to generate higher load (causes more overloads)
        cpu_range = cfg("simulation.workload.base_cpu_range", [30, 90])
        ram_range = cfg("simulation.workload.base_ram_range", [25, 75])
        net_range = cfg("simulation.workload.base_network_range", [10, 60])

        self.cpu = cpu if cpu is not None else random.uniform(*cpu_range)
        self.ram = ram if ram is not None else random.uniform(*ram_range)
        self.network = network if network is not None else random.uniform(*net_range)
        
        # Apply initial burst spike with 25% probability
        if random.random() < 0.25:
            self.cpu += random.uniform(20, 40)
            self.cpu = min(100, self.cpu)

        self.workload_pattern = cfg("simulation.workload.pattern", "time_series")

        # Internal state
        self.tick = 0
        self._phase_offset = random.uniform(0, 2 * math.pi)

        # Burst state
        self.burst_remaining = 0
        self.burst_cpu_extra = 0.0

    # ------------------------------------------------------------------ #
    #  Update methods
    # ------------------------------------------------------------------ #
    def update_usage(self):
        """Update resource usage based on configured workload pattern."""
        self.tick += 1

        if self.workload_pattern == "time_series":
            self._update_time_series()
        elif self.workload_pattern == "burst":
            self._update_burst()
        else:
            self._update_random()

        # Clamp values
        self.cpu = max(0, min(100, self.cpu))
        self.ram = max(0, min(100, self.ram))
        self.network = max(0, min(100, self.network))

    def _update_random(self):
        """Enhanced random fluctuation model with burst spikes."""
        self.cpu += random.uniform(-10, 25)
        self.ram += random.uniform(-8, 18)
        self.network += random.uniform(-8, 15)
        
        # Random burst spike with 25% probability
        if random.random() < 0.25:
            self.cpu += random.uniform(20, 40)

    def _update_time_series(self):
        """Sinusoidal diurnal pattern with Gaussian noise and occasional micro-bursts."""
        ts_cfg = cfg("simulation.workload.time_series", {})
        cpu_amp = ts_cfg.get("cpu_amplitude", 25.0)
        cpu_period = ts_cfg.get("cpu_period", 24)
        ram_amp = ts_cfg.get("ram_amplitude", 15.0)
        ram_period = ts_cfg.get("ram_period", 48)
        noise_std = ts_cfg.get("noise_stddev", 5.0)

        base_cpu = cfg("simulation.workload.base_cpu_range", [10, 70])
        base_ram = cfg("simulation.workload.base_ram_range", [15, 65])
        mid_cpu = sum(base_cpu) / 2
        mid_ram = sum(base_ram) / 2

        self.cpu = mid_cpu + cpu_amp * math.sin(
            2 * math.pi * self.tick / cpu_period + self._phase_offset
        ) + np.random.normal(0, noise_std)

        self.ram = mid_ram + ram_amp * math.sin(
            2 * math.pi * self.tick / ram_period + self._phase_offset
        ) + np.random.normal(0, noise_std * 0.7)

        self.network += random.uniform(-8, 10) + np.random.normal(0, noise_std * 0.4)

        # Chance of a random micro-burst (increased probability for more overloads)
        burst_cfg = cfg("simulation.workload.burst", {})
        burst_prob = burst_cfg.get("probability", 0.15)  # Increased from 0.05
        if random.random() < burst_prob:
            spike_range = burst_cfg.get("cpu_spike_range", [25, 45])
            self.cpu += random.uniform(*spike_range)

    def _update_burst(self):
        """Burst traffic model: sudden spike + gradual cooldown."""
        burst_cfg = cfg("simulation.workload.burst", {})
        burst_prob = burst_cfg.get("probability", 0.05)
        spike_range = burst_cfg.get("cpu_spike_range", [30, 50])
        duration = burst_cfg.get("duration_ticks", 3)
        cooldown = burst_cfg.get("cooldown_rate", 0.7)

        if self.burst_remaining <= 0 and random.random() < burst_prob:
            self.burst_remaining = duration
            self.burst_cpu_extra = random.uniform(*spike_range)

        if self.burst_remaining > 0:
            self.cpu += self.burst_cpu_extra
            self.burst_remaining -= 1
            self.burst_cpu_extra *= cooldown
        else:
            self.cpu += random.uniform(-10, 12)

        self.ram += random.uniform(-8, 10)
        self.network += random.uniform(-8, 10)

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #
    def get_metrics(self):
        """Return current VM metrics as a dictionary."""
        return {
            "vm_id": self.vm_id,
            "host_id": self.host_id,
            "cpu": round(self.cpu, 2),
            "ram": round(self.ram, 2),
            "network": round(self.network, 2),
        }

    def __repr__(self):
        return (f"VM({self.vm_id}, host={self.host_id}, "
                f"cpu={self.cpu:.1f}%, ram={self.ram:.1f}%, net={self.network:.1f}%)")
