"""
data_generator.py - CSV Data Writer
======================================
Saves simulated VM and host metrics to CSV files for ML training.
"""

import os
import csv
from logger import setup_logger

logger = setup_logger(__name__)


def save_metrics_to_csv(records, output_path="data/vm_metrics.csv"):
    """
    Save a list of metric records to a CSV file.
    
    Dynamically infers fieldnames from the records to support varying schemas.
    
    Args:
        records (list[dict]): List of metric dictionaries.
        output_path (str): Path to the output CSV file.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Safety check: empty records
    if not records:
        logger.warning(f"[DATA] No records to save. Skipping CSV write to '{output_path}'")
        print(f"[DATA] Warning: No records to save. Skipping CSV write.")
        return False
    
    # Safety check: records must be dictionaries
    if not isinstance(records[0], dict):
        logger.error(f"[DATA] Records must be dictionaries, got {type(records[0])}")
        print(f"[DATA] Error: Records must be dictionaries.")
        return False

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Dynamically infer fieldnames from the first record
    fieldnames = list(records[0].keys())
    
    # Log the detected fields
    logger.info(f"[DATA] Detected {len(fieldnames)} fields: {fieldnames}")

    try:
        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"[DATA] Saved {len(records)} records to '{output_path}'")
        print(f"[DATA] Saved {len(records)} records to '{output_path}'")
        return True
        
    except Exception as e:
        logger.error(f"[DATA] Failed to save CSV: {e}")
        print(f"[DATA] Error saving CSV: {e}")
        return False


def append_metrics_to_csv(records, output_path="data/vm_metrics.csv"):
    """
    Append metric records to an existing CSV file (creates if not exists).
    
    Dynamically infers fieldnames from the records.
    
    Args:
        records (list[dict]): List of metric dictionaries to append.
        output_path (str): Path to the CSV file.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Safety check: empty records
    if not records:
        logger.warning(f"[DATA] No records to append. Skipping.")
        print(f"[DATA] Warning: No records to append.")
        return False
    
    # Safety check: records must be dictionaries
    if not isinstance(records[0], dict):
        logger.error(f"[DATA] Records must be dictionaries, got {type(records[0])}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Dynamically infer fieldnames from records
    fieldnames = list(records[0].keys())
    
    file_exists = os.path.exists(output_path)

    try:
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)
        
        logger.info(f"[DATA] Appended {len(records)} records to '{output_path}'")
        return True
        
    except Exception as e:
        logger.error(f"[DATA] Failed to append CSV: {e}")
        return False
