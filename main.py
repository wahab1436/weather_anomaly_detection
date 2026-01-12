#!/usr/bin/env python3
"""
Main orchestration script for the Weather Anomaly Detection System.
Runs the complete pipeline: scraping → preprocessing → ML → dashboard.
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import schedule
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraping.scrape_weather_alerts import main as scrape_main
from preprocessing.preprocess_text import preprocess_pipeline
from ml.anomaly_detection import run_anomaly_detection
from ml.forecast_model import run_forecasting
from utils.helpers import setup_logging, generate_plain_english_insights, save_to_json

# Setup logging
setup_logging('logs/system.log')
logger = logging.getLogger(__name__)

# Import pandas late to avoid early import issues
import pandas as pd

# ------------------------------
# WeatherAnomalySystem class
# ------------------------------
class WeatherAnomalySystem:
    """Main orchestration class for the weather anomaly detection system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the system with configuration."""
        self.config = self.load_config(config_path)
        self.running = False
        
        # File paths
        self.raw_data_path = "data/raw/weather_alerts_raw.csv"
        self.processed_data_path = "data/processed/weather_alerts_processed.csv"
        self.daily_data_path = "data/processed/weather_alerts_daily.csv"
        self.anomaly_model_path = "models/isolation_forest.pkl"
        self.forecast_model_path = "models/xgboost_forecast.pkl"
        self.anomaly_output_path = "data/output/anomaly_results.csv"
        self.forecast_output_path = "data/output/forecast_results.csv"

    # ---------- Methods (same as your existing code) ----------
    # load_config, run_scraping, run_preprocessing, run_anomaly_detection_pipeline,
    # run_forecasting_pipeline, generate_insights, run_complete_pipeline,
    # cleanup_old_data, schedule_jobs, run_scheduler remain unchanged.
    # ------------------------------

    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        if not self.config['enable_dashboard']:
            logger.info("Dashboard disabled in configuration")
            return
        
        logger.info(f"Starting dashboard on port {self.config['dashboard_port']}...")
        
        # Run Streamlit dashboard from this main entry point
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
            "--server.port", str(self.config['dashboard_port'])
        ])

# ------------------------------
# Main function
# ------------------------------
def main():
    """Main entry point for the system."""

    # Detect if Streamlit is running
    if 'streamlit' in sys.argv[0].lower():
        system = WeatherAnomalySystem()
        system.start_dashboard()
        return

    # ---------- CLI argument parsing ----------
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Weather Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run pipeline      # Run the complete pipeline once
  %(prog)s run scheduler     # Run with scheduled jobs
  %(prog)s scrape            # Run only scraping
  %(prog)s preprocess        # Run only preprocessing
  %(prog)s detect-anomalies  # Run only anomaly detection
  %(prog)s forecast          # Run only forecasting
        """
    )

    parser.add_argument(
        'command',
        choices=[
            'run', 'pipeline', 'scheduler',
            'scrape', 'preprocess', 'detect-anomalies', 'forecast',
            'dashboard', 'cleanup', 'insights'
        ],
        help='Command to execute'
    )

    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--mode',
        choices=['once', 'continuous'],
        default='once',
        help='Execution mode'
    )

    args = parser.parse_args()

    # Initialize system
    system = WeatherAnomalySystem(args.config)

    # Execute command
    if args.command in ['run', 'pipeline']:
        system.run_complete_pipeline()
    
    elif args.command == 'scheduler':
        system.run_scheduler()
    
    elif args.command == 'scrape':
        system.run_scraping()
    
    elif args.command == 'preprocess':
        system.run_preprocessing()
    
    elif args.command == 'detect-anomalies':
        system.run_anomaly_detection_pipeline()
    
    elif args.command == 'forecast':
        system.run_forecasting_pipeline()
    
    elif args.command == 'dashboard':
        system.start_dashboard()
    
    elif args.command == 'cleanup':
        system.cleanup_old_data()
    
    elif args.command == 'insights':
        system.generate_insights()
    
    else:
        parser.print_help()

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()
