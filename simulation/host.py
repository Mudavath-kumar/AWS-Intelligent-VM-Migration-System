"""
host.py - Physical Host Simulation (Enhanced)
================================================
Represents a physical host (server) with heterogeneous capacity.
Supports configurable CPU/RAM capacity and dynamic power model.
"""

from config import get as cfg


class Host:
    """
    Simulates a physical host server running multiple VMs.

    Attributes:
        host_id (str): Unique identifier for the host.
        vms (list): List of VM objects assigned to this host.
        cpu_capacity (float): Maximum CPU capacity.
        ram_capacity (float): Maximum RAM capacity.
        overload_threshold (float): CPU threshold to declare overload.
        p_idle (float): Idle power consumption (watts).
        p_max (float): Max power consumption (watts).
    """

    def __init__(self, host_id, cpu_capacity=100, ram_capacity=100, overload_threshold=85):
        self.host_id = host_id
        self.vms = []
        self.cpu_capacity = cpu_capacity
        self.ram_capacity = ram_capacity
        self.overload_threshold = overload_threshold

        # Dynamic power model parameters from config
        self.p_idle = cfg("evaluation.energy.p_idle", 100)
        self.p_max = cfg("evaluation.energy.p_max", 300)

    def add_vm(self, vm):
        """Add a VM to this host."""
        vm.host_id = self.host_id
        self.vms.append(vm)

    def remove_vm(self, vm_id):
        """Remove a VM from this host by its ID."""
        for i, vm in enumerate(self.vms):
            if vm.vm_id == vm_id:
                return self.vms.pop(i)
        return None

    def get_total_cpu(self):
        """Average CPU utilization across all VMs (0-100%)."""
        if not self.vms:
            return 0.0
        return sum(vm.cpu for vm in self.vms) / len(self.vms)

    def get_total_ram(self):
        """Average RAM utilization across all VMs (0-100%)."""
        if not self.vms:
            return 0.0
        return sum(vm.ram for vm in self.vms) / len(self.vms)

    def get_total_network(self):
        """Average network utilization across all VMs (0-100%)."""
        if not self.vms:
            return 0.0
        return sum(vm.network for vm in self.vms) / len(self.vms)

    def get_utilization(self):
        """
        Get normalized utilization factor (0.0 - 1.0) relative to capacity.
        Accounts for heterogeneous CPU capacity.
        """
        if not self.vms:
            return 0.0
        total_cpu_demand = sum(vm.cpu for vm in self.vms)
        return min(1.0, total_cpu_demand / (self.cpu_capacity * len(self.vms) / 100.0 * len(self.vms)))

    def is_overloaded(self):
        """Check if the host is overloaded (CPU > threshold)."""
        return self.get_total_cpu() > self.overload_threshold

    def get_power_consumption(self):
        """
        Dynamic power model:
            Power = P_idle + (P_max - P_idle) * utilization
        
        Returns:
            float: Power consumption in watts.
        """
        if not self.vms:
            return self.p_idle * 0.3   # idle host

        utilization = self.get_total_cpu() / 100.0
        return self.p_idle + (self.p_max - self.p_idle) * utilization

    def get_metrics(self):
        """Return host-level aggregated metrics."""
        return {
            "host_id": self.host_id,
            "num_vms": len(self.vms),
            "cpu_capacity": self.cpu_capacity,
            "ram_capacity": self.ram_capacity,
            "total_cpu": round(self.get_total_cpu(), 2),
            "total_ram": round(self.get_total_ram(), 2),
            "total_network": round(self.get_total_network(), 2),
            "overloaded": self.is_overloaded(),
            "power_watts": round(self.get_power_consumption(), 2),
        }

    def __repr__(self):
        return (f"Host({self.host_id}, cap={self.cpu_capacity}, VMs={len(self.vms)}, "
                f"CPU={self.get_total_cpu():.1f}%, "
                f"overloaded={self.is_overloaded()})")
