"""
Utility functions for the weather anomaly detection system.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from functools import wraps
import time

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "logs/weather_dashboard.log"):
    """Setup comprehensive logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set higher level for libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate dataframe has required columns and data."""
    if df.empty:
        logger.warning("DataFrame is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for excessive null values
    null_counts = df[required_columns].isnull().sum()
    total_rows = len(df)
    
    for col, null_count in null_counts.items():
        null_percentage = (null_count / total_rows) * 100
        if null_percentage > 50:  # More than 50% null
            logger.warning(f"Column {col} has {null_percentage:.1f}% null values")
    
    return True

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics for the dataset."""
    stats = {
        'total_records': len(df),
        'date_range': {},
        'alert_types': {},
        'missing_values': {}
    }
    
    # Date range
    if 'issued_date' in df.columns:
        df['issued_date'] = pd.to_datetime(df['issued_date'], errors='coerce')
        valid_dates = df['issued_date'].dropna()
        if not valid_dates.empty:
            stats['date_range'] = {
                'start': valid_dates.min().strftime('%Y-%m-%d'),
                'end': valid_dates.max().strftime('%Y-%m-%d'),
                'days': (valid_dates.max() - valid_dates.min()).days
            }
    
    # Alert type distribution
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        stats['alert_types'] = type_counts.to_dict()
    
    # Missing values
    stats['missing_values'] = df.isnull().sum().to_dict()
    
    return stats

def generate_plain_english_insights(
    daily_stats: pd.DataFrame,
    anomalies: pd.DataFrame,
    forecasts: pd.DataFrame
) -> List[str]:
    """Generate plain English insights from analysis results."""
    insights = []
    
    if daily_stats.empty:
        insights.append("No historical data available for analysis.")
        return insights
    
    try:
        # Ensure total_alerts column exists
        if 'total_alerts' not in daily_stats.columns:
            insights.append("System operational. Monitoring weather alerts for anomalies.")
            return insights
        
        # Recent activity
        recent_days = daily_stats.tail(7)
        if not recent_days.empty and 'total_alerts' in recent_days.columns:
            avg_recent = recent_days['total_alerts'].mean()
            prev_week = daily_stats.iloc[-14:-7]['total_alerts'].mean() if len(daily_stats) >= 14 else 0
            
            if avg_recent > prev_week * 1.5 and prev_week > 0:
                insights.append(f"Alert activity has increased significantly in the past week (average {avg_recent:.1f} alerts/day vs {prev_week:.1f} previously).")
            elif avg_recent < prev_week * 0.7 and prev_week > 0:
                insights.append(f"Alert activity has decreased in the past week (average {avg_recent:.1f} alerts/day vs {prev_week:.1f} previously).")
            else:
                insights.append(f"Alert activity remains stable at around {avg_recent:.1f} alerts per day.")
        
        # Anomaly insights
        if not anomalies.empty and 'is_anomaly' in anomalies.columns:
            # Ensure issued_date is datetime if it exists
            if 'issued_date' in anomalies.columns:
                anomalies = anomalies.copy()
                anomalies['issued_date'] = pd.to_datetime(anomalies['issued_date'], errors='coerce')
                
                # Set as index if not already
                if not isinstance(anomalies.index, pd.DatetimeIndex):
                    anomalies = anomalies.set_index('issued_date')
            
            recent_anomalies = anomalies.tail(30)
            anomaly_count = recent_anomalies['is_anomaly'].sum()
            
            if anomaly_count > 0:
                anomaly_dates = recent_anomalies[recent_anomalies['is_anomaly']]
                if not anomaly_dates.empty:
                    if isinstance(anomaly_dates.index, pd.DatetimeIndex):
                        last_anomaly_date = anomaly_dates.index[-1]
                        insights.append(f"Detected {anomaly_count} anomalous days in the past month, most recently on {last_anomaly_date.strftime('%B %d')}.")
                    else:
                        insights.append(f"Detected {anomaly_count} anomalous days in the past month.")
                
                # High severity anomalies
                if 'anomaly_severity' in recent_anomalies.columns:
                    high_severity = recent_anomalies[
                        (recent_anomalies['is_anomaly']) & 
                        (recent_anomalies['anomaly_severity'].isin(['high', 'critical']))
                    ]
                    
                    if not high_severity.empty:
                        insights.append(f"Found {len(high_severity)} high-severity anomalies requiring attention.")
        
        # Forecast insights
        if not forecasts.empty and 'target' in forecasts.columns and 'forecast' in forecasts.columns:
            latest_forecast = forecasts[forecasts['target'] == 'total_alerts'].tail(7)
            
            if not latest_forecast.empty and 'total_alerts' in daily_stats.columns:
                avg_forecast = latest_forecast['forecast'].mean()
                current_avg = daily_stats['total_alerts'].tail(7).mean()
                
                if avg_forecast > current_avg * 1.2:
                    insights.append(f"Forecast predicts increased alert activity in the coming week (average {avg_forecast:.1f} alerts/day expected).")
                elif avg_forecast < current_avg * 0.8:
                    insights.append(f"Forecast predicts decreased alert activity in the coming week (average {avg_forecast:.1f} alerts/day expected).")
        
        # Alert type distribution
        alert_type_cols = [col for col in daily_stats.columns if col in [
            'flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold'
        ]]
        
        if alert_type_cols:
            recent_types = daily_stats[alert_type_cols].tail(7).sum()
            if recent_types.sum() > 0:
                dominant_type = recent_types.idxmax()
                dominant_count = recent_types.max()
                
                if dominant_count > 0:
                    insights.append(f"Most frequent alert type recently: {dominant_type.capitalize()} alerts.")
        
        # Seasonal patterns
        if 'month' in daily_stats.columns and 'total_alerts' in daily_stats.columns and len(daily_stats) > 365:
            current_month = datetime.now().month
            monthly_avg = daily_stats.groupby('month')['total_alerts'].mean()
            
            if current_month in monthly_avg.index and not monthly_avg.empty:
                current_month_avg = monthly_avg[current_month]
                yearly_avg = monthly_avg.mean()
                
                if current_month_avg > yearly_avg * 1.3:
                    insights.append(f"Current month typically sees higher than average alert activity ({current_month_avg:.1f} vs yearly average {yearly_avg:.1f}).")
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}", exc_info=True)
        insights.append("Analysis complete. Review dashboard for detailed metrics.")
    
    # Ensure we always have at least one insight
    if not insights:
        insights.append("System operational. Monitoring weather alerts for anomalies.")
    
    return insights

def format_number(value: float) -> str:
    """Format numbers for display."""
    if pd.isna(value):
        return "N/A"
    
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    elif value == 0:
        return "0"
    elif abs(value) < 0.01:
        return f"{value:.2e}"
    elif abs(value) < 1:
        return f"{value:.3f}"
    elif abs(value) < 100:
        return f"{value:.1f}"
    else:
        return f"{value:.0f}"

def calculate_performance_change(current: float, previous: float) -> Dict[str, Any]:
    """Calculate performance change metrics."""
    if previous == 0:
        return {
            'change': 0,
            'percentage': 0,
            'direction': 'neutral',
            'significant': False
        }
    
    change = current - previous
    percentage = (change / previous) * 100
    
    direction = 'increase' if change > 0 else 'decrease' if change < 0 else 'neutral'
    significant = abs(percentage) > 10  # More than 10% change is significant
    
    return {
        'change': change,
        'percentage': abs(percentage),
        'direction': direction,
        'significant': significant
    }

def retry_on_failure(max_retries: int = 3, delay: int = 5):
    """Decorator for retrying functions on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise
            return None
        return wrapper
    return decorator

def save_to_json(data: Any, filepath: str):
    """Save data to JSON file with proper directory creation."""
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise

def load_from_json(filepath: str) -> Any:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {str(e)}")
        return {}

def get_data_last_updated(filepath: str) -> Optional[datetime]:
    """Get when data was last updated."""
    try:
        if os.path.exists(filepath):
            mtime = os.path.getmtime(filepath)
            return datetime.fromtimestamp(mtime)
    except Exception as e:
        logger.error(f"Error getting last updated time for {filepath}: {str(e)}")
    
    return None

def create_backup(filepath: str, backup_dir: str = "backups"):
    """Create a backup of a file."""
    if not os.path.exists(filepath):
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
    
    try:
        import shutil
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")

def cleanup_old_files(directory: str, days_to_keep: int = 30, pattern: str = "*.csv"):
    """Clean up old files in a directory."""
    import glob
    
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    files_removed = 0
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < cutoff_time:
                os.remove(filepath)
                logger.info(f"Removed old file: {filepath}")
                files_removed += 1
        except Exception as e:
            logger.error(f"Error removing file {filepath}: {str(e)}")
    
    logger.info(f"Cleanup complete. Removed {files_removed} files from {directory}")

if __name__ == "__main__":
    # Test the helper functions
    setup_logging()
    
    # Create a test dataframe
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'value': range(10)
    })
    
    stats = calculate_statistics(test_df)
    print(json.dumps(stats, indent=2))
