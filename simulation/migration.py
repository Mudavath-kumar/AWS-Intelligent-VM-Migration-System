"""
migration.py - VM Migration Logic
====================================
Handles the actual migration of VMs between hosts.
Logs migration events and displays before/after status.
"""

import datetime


# Global migration log
migration_log = []


def migrate_vm(source_host, target_host, vm_id):
    """
    Migrate a VM from a source host to a target host.
    
    Steps:
        1. Print before-migration status
        2. Remove VM from source host
        3. Add VM to target host
        4. Update VM's host_id
        5. Log the migration event
        6. Print after-migration status
    
    Args:
        source_host (Host): The host to migrate FROM.
        target_host (Host): The host to migrate TO.
        vm_id (str): The ID of the VM to migrate.
        
    Returns:
        dict: Migration event record, or None if VM not found.
    """
    # ---- Before Migration Status ----
    print(f"\n{'-' * 50}")
    print(f"BEFORE MIGRATION")
    print(f"  Source: {source_host.host_id} -- CPU: {source_host.get_total_cpu():.1f}% "
          f"-- VMs: {len(source_host.vms)}")
    print(f"  Target: {target_host.host_id} -- CPU: {target_host.get_total_cpu():.1f}% "
          f"-- VMs: {len(target_host.vms)}")

    # ---- Perform Migration ----
    vm = source_host.remove_vm(vm_id)
    if vm is None:
        print(f"  [ERROR] VM {vm_id} not found on {source_host.host_id}")
        return None

    target_host.add_vm(vm)

    # ---- Log Migration Event ----
    event = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vm_id": vm_id,
        "source_host": source_host.host_id,
        "target_host": target_host.host_id,
        "source_cpu_after": round(source_host.get_total_cpu(), 2),
        "target_cpu_after": round(target_host.get_total_cpu(), 2),
        "vm_cpu": round(vm.cpu, 2),
    }
    migration_log.append(event)

    # ---- After Migration Status ----
    print(f"\nAFTER MIGRATION")
    print(f"  [OK] Migrated {vm_id} from {source_host.host_id} -> {target_host.host_id}")
    print(f"  Source: {source_host.host_id} -- CPU: {source_host.get_total_cpu():.1f}% "
          f"-- VMs: {len(source_host.vms)}")
    print(f"  Target: {target_host.host_id} -- CPU: {target_host.get_total_cpu():.1f}% "
          f"-- VMs: {len(target_host.vms)}")
    print(f"{'-' * 50}")

    return event


def get_migration_log():
    """
    Return the full migration log.
    
    Returns:
        list[dict]: All migration events.
    """
    return migration_log


def clear_migration_log():
    """Clear the migration log."""
    global migration_log
    migration_log = []


def print_migration_summary():
    """Print a summary of all recorded migrations."""
    print(f"\n{'=' * 60}")
    print(f"MIGRATION SUMMARY -- Total Migrations: {len(migration_log)}")
    print(f"{'=' * 60}")
    for i, event in enumerate(migration_log, 1):
        print(f"  {i}. [{event['timestamp']}] {event['vm_id']}: "
              f"{event['source_host']} -> {event['target_host']} "
              f"(VM CPU: {event['vm_cpu']}%)")
    print(f"{'=' * 60}")
