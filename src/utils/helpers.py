"""
Helper utility functions.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str = 'config.yaml'):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


def ensure_dir(directory: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_data_path(base_dir: str, data_type: str = 'raw') -> Path:
    """
    Get path to data directory.
    
    Args:
        base_dir: Base project directory
        data_type: Type of data ('raw', 'processed', 'external')
        
    Returns:
        Path to data directory
    """
    return Path(base_dir) / 'data' / data_type
