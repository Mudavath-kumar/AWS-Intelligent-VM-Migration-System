"""
config.py - Configuration Manager
====================================
Loads experiment configuration from config.yaml.
Provides a global Config object accessible throughout the project.
"""

import os
import yaml


_CONFIG = None
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(config_path=None):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to config.yaml. Uses default if None.

    Returns:
        dict: Configuration dictionary.
    """
    global _CONFIG
    path = config_path or _CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config():
    """
    Get the loaded configuration. Loads from default path if not loaded yet.

    Returns:
        dict: Configuration dictionary.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get(key_path, default=None):
    """
    Get a nested config value using dot-separated key path.

    Example:
        get("simulation.num_hosts")  -> 5
        get("model.lstm.epochs")     -> 50

    Args:
        key_path (str): Dot-separated path to config value.
        default: Default value if key not found.

    Returns:
        The configuration value.
    """
    cfg = get_config()
    keys = key_path.split(".")
    for key in keys:
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg
