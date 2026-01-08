"""
Helper utility functions for the weather anomaly detection system
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
from typing import Any, Dict, Optional

def setup_logging(log_file: str = "logs/weather_detection.log", level: str = "INFO"):
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def create_directories(base_path: str = "."):
    """
    Create all necessary directories for the project
    
    Args:
        base_path: Base path for creating directories
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/output",
        "models",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Dictionary with configuration
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Return default configuration
        return {
            "scraping": {
                "interval": 3600,
                "user_agent": "WeatherAnomalyDetection/1.0"
            },
            "ml": {
                "anomaly_contamination": 0.05,
                "forecast_horizon": 7
            },
            "dashboard": {
                "port": 8501,
                "host": "0.0.0.0"
            }
        }

def save_data(data: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save DataFrame to file with proper error handling
    
    Args:
        data: DataFrame to save
        filepath: Path to save file
        index: Whether to save index
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save based on file extension
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=index)
        elif filepath.endswith('.parquet'):
            data.to_parquet(filepath, index=index)
        elif filepath.endswith('.json'):
            data.to_json(filepath, orient='records')
        else:
            data.to_csv(filepath, index=index)
        
        print(f"Data saved to {filepath} ({len(data)} rows)")
        return True
        
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return False

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load data from file with proper error handling
    
    Args:
        filepath: Path to load file from
    
    Returns:
        Loaded DataFrame or None if error
    """
    try:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        # Load based on file extension
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        elif filepath.endswith('.json'):
            data = pd.read_json(filepath, orient='records')
        else:
            data = pd.read_csv(filepath)
        
        print(f"Data loaded from {filepath} ({len(data)} rows)")
        return data
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def get_last_updated_time(filepath: str) -> Optional[datetime]:
    """
    Get the last modified time of a file
    
    Args:
        filepath: Path to file
    
    Returns:
        Last modified datetime or None if file doesn't exist
    """
    if os.path.exists(filepath):
        timestamp = os.path.getmtime(filepath)
        return datetime.fromtimestamp(timestamp)
    return None

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate DataFrame structure
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
    
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

def clean_memory():
    """
    Clean up memory by forcing garbage collection
    """
    import gc
    gc.collect()
