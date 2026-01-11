import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/raw/backups',
        'data/processed',
        'data/output',
        'models',
        'notebooks',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Validate DataFrame structure."""
    if df.empty:
        logger.warning("DataFrame is empty")
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return False
    
    return True

def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for alert data."""
    stats = {}
    
    if df.empty:
        return stats
    
    # Basic counts
    stats['total_alerts'] = len(df)
    stats['unique_regions'] = df['region'].nunique() if 'region' in df.columns else 0
    stats['unique_alert_types'] = df['alert_type'].nunique() if 'alert_type' in df.columns else 0
    
    # Date range
    if 'scraped_at' in df.columns:
        dates = pd.to_datetime(df['scraped_at'])
        stats['date_range'] = {
            'start': dates.min().strftime('%Y-%m-%d'),
            'end': dates.max().strftime('%Y-%m-%d'),
            'days': (dates.max() - dates.min()).days + 1
        }
    
    # Alert type distribution
    if 'alert_type' in df.columns:
        alert_counts = df['alert_type'].value_counts().head(10).to_dict()
        stats['top_alert_types'] = alert_counts
    
    # Region distribution
    if 'region' in df.columns:
        region_counts = df['region'].value_counts().head(10).to_dict()
        stats['top_regions'] = region_counts
    
    # Time-based patterns
    if 'scraped_at' in df.columns:
        df['hour'] = pd.to_datetime(df['scraped_at']).dt.hour
        hourly_counts = df.groupby('hour').size()
        stats['peak_hour'] = int(hourly_counts.idxmax()) if not hourly_counts.empty else 0
    
    return stats

def generate_timestamp() -> str:
    """Generate current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file safely."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {str(e)}")
    return None

def save_json_file(data: Dict, filepath: str):
    """Save data to JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {str(e)}")

def format_insight_text(insight_type: str, data: Dict) -> str:
    """Format insights into plain English text."""
    if insight_type == 'anomaly':
        if not data.get('key_anomalies'):
            return data.get('summary', 'No anomalies detected.')
        
        text = data.get('summary', '') + "\n\n"
        for anomaly in data.get('key_anomalies', [])[:3]:  # Top 3 anomalies
            text += f"• On {anomaly.get('date', 'unknown date')}: {anomaly.get('reason', 'Unusual pattern')}\n"
        
        if data.get('recommendations'):
            text += "\nRecommendations:\n"
            for rec in data.get('recommendations', []):
                text += f"• {rec}\n"
        
        return text
    
    elif insight_type == 'forecast':
        if not data.get('key_forecasts'):
            return data.get('summary', 'No forecast available.')
        
        text = data.get('summary', '') + "\n\nNext 7-day forecast:\n"
        for forecast in data.get('key_forecasts', []):
            text += f"• {forecast.get('date')}: {forecast.get('predicted_alerts')} alerts (range: {forecast.get('range')})\n"
        
        if data.get('recommendations'):
            text += "\nRecommendations:\n"
            for rec in data.get('recommendations', []):
                text += f"• {rec}\n"
        
        return text
    
    return str(data)

def check_data_freshness(filepath: str, max_age_hours: int = 2) -> bool:
    """Check if data file is fresh."""
    if not os.path.exists(filepath):
        return False
    
    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    
    return age_hours <= max_age_hours

def cleanup_old_files(directory: str, pattern: str, max_age_days: int = 30):
    """Clean up old files from directory."""
    try:
        now = datetime.now()
        for filename in os.listdir(directory):
            if pattern in filename:
                filepath = os.path.join(directory, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (now - file_mtime).days
                
                if age_days > max_age_days:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to cleanup files: {str(e)}")
