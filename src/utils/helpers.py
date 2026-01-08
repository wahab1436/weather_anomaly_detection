"""
Helper utility functions for the weather anomaly detection system
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
import pickle
import gzip
import hashlib

def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
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
    logger.info(f"Logging setup complete. Level: {level}")
    return logger

def create_directories() -> None:
    """Create all necessary directories for the project"""
    directories = [
        "data/raw",
        "data/processed",
        "data/output",
        "models",
        "logs",
        "notebooks",
        "src/scraping",
        "src/preprocessing",
        "src/ml",
        "src/utils",
        "src/dashboard"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Created all project directories")

def load_data(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load data from file with proper error handling
    
    Args:
        filepath: Path to load file from
        **kwargs: Additional arguments for pandas read functions
    
    Returns:
        Loaded DataFrame or None if error
    """
    try:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        # Determine file type and load accordingly
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath, **kwargs)
        elif filepath.endswith('.json'):
            data = pd.read_json(filepath, **kwargs)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            # Try CSV as default
            data = pd.read_csv(filepath, **kwargs)
        
        print(f"Loaded data from {filepath} ({len(data)} rows)")
        return data
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {str(e)}")
        return None

def save_data(data: pd.DataFrame, filepath: str, **kwargs) -> bool:
    """
    Save DataFrame to file with proper error handling
    
    Args:
        data: DataFrame to save
        filepath: Path to save file
        **kwargs: Additional arguments for pandas write functions
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine file type and save accordingly
        if filepath.endswith('.csv'):
            data.to_csv(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            data.to_parquet(filepath, **kwargs)
        elif filepath.endswith('.json'):
            data.to_json(filepath, **kwargs)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        else:
            # Save as CSV by default
            data.to_csv(filepath, **kwargs)
        
        print(f"Saved data to {filepath} ({len(data)} rows)")
        return True
        
    except Exception as e:
        print(f"Error saving data to {filepath}: {str(e)}")
        return False

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate DataFrame structure and quality
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'issues': []
    }
    
    if df.empty:
        results['is_valid'] = False
        results['issues'].append('DataFrame is empty')
        return results
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['is_valid'] = False
            results['issues'].append(f'Missing required columns: {missing_cols}')
    
    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            results['missing_values'][col] = missing_count
            results['issues'].append(f'Column {col} has {missing_count} missing values')
        
        # Record data types
        results['data_types'][col] = str(df[col].dtype)
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        results['issues'].append(f'Found {duplicate_count} duplicate rows')
    
    return results

def clean_memory() -> None:
    """Clean up memory by forcing garbage collection"""
    import gc
    gc.collect()

def get_file_hash(filepath: str) -> Optional[str]:
    """
    Calculate MD5 hash of a file
    
    Args:
        filepath: Path to file
    
    Returns:
        MD5 hash string or None if error
    """
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {str(e)}")
        return None

def compress_file(input_path: str, output_path: str = None) -> bool:
    """
    Compress a file using gzip
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if output_path is None:
            output_path = input_path + '.gz'
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"Compressed {input_path} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error compressing file {input_path}: {str(e)}")
        return False

def decompress_file(input_path: str, output_path: str = None) -> bool:
    """
    Decompress a gzipped file
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if output_path is None:
            output_path = input_path.replace('.gz', '')
        
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"Decompressed {input_path} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error decompressing file {input_path}: {str(e)}")
        return False

def format_timestamp(timestamp: Any, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Timestamp to format
        format_str: Format string
    
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        
        return timestamp.strftime(format_str)
    except Exception:
        return str(timestamp)

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': [],
        'categorical_columns': [],
        'missing_values_total': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Analyze each column
    for col in df.columns:
        col_stats = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'unique': df[col].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'min': float(df[col].min()) if df[col].notna().any() else None,
                'max': float(df[col].max()) if df[col].notna().any() else None,
                'mean': float(df[col].mean()) if df[col].notna().any() else None,
                'std': float(df[col].std()) if df[col].notna().any() else None
            })
            stats['numeric_columns'].append(col_stats)
        else:
            # For categorical/text columns
            if df[col].notna().any():
                col_stats['top_value'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
                col_stats['top_frequency'] = int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
            stats['categorical_columns'].append(col_stats)
    
    return stats
