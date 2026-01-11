#!/usr/bin/env python3
"""
Main pipeline orchestrator for Weather Anomaly Detection System.
Run this script to execute the complete data pipeline.
"""

import logging
import schedule
import time
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scraping.scrape_weather_alerts import main as scrape_main
from preprocessing.preprocess_text import main as preprocess_main
from ml.anomaly_detection import main as anomaly_main
from ml.forecast_model import main as forecast_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete data pipeline."""
    logger.info("Starting complete pipeline execution")
    
    try:
        # Step 1: Scrape data
        logger.info("Step 1: Scraping weather alerts")
        scrape_main()
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data")
        preprocess_main()
        
        # Step 3: Detect anomalies
        logger.info("Step 3: Detecting anomalies")
        anomaly_main()
        
        # Step 4: Generate forecasts
        logger.info("Step 4: Generating forecasts")
        forecast_main()
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

def run_hourly():
    """Run pipeline hourly."""
    run_complete_pipeline()

def run_daily():
    """Run pipeline daily (for more intensive tasks)."""
    logger.info("Running daily maintenance tasks")
    # Additional daily tasks can be added here

def main():
    """Main function with scheduling options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weather Anomaly Detection Pipeline')
    parser.add_argument('--run-once', action='store_true', help='Run pipeline once and exit')
    parser.add_argument('--schedule', action='store_true', help='Schedule pipeline to run hourly')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/raw/backups', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.run_once:
        logger.info("Running pipeline once")
        run_complete_pipeline()
        
    elif args.schedule:
        logger.info("Starting scheduled pipeline (hourly)")
        
        # Schedule hourly runs
        schedule.every().hour.at(":00").do(run_hourly)
        
        # Schedule daily maintenance at 2 AM
        schedule.every().day.at("02:00").do(run_daily)
        
        # Run once immediately
        run_complete_pipeline()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("WEATHER ANOMALY DETECTION PIPELINE")
        print("="*60)
        print("\nOptions:")
        print("1. Run pipeline once")
        print("2. Start scheduled pipeline (hourly)")
        print("3. Start dashboard")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            run_complete_pipeline()
        elif choice == '2':
            # Run in background
            print("Starting scheduled pipeline in background...")
            import subprocess
            subprocess.Popen([sys.executable, __file__, '--schedule'])
            print("Scheduled pipeline started. Check logs/pipeline.log for output.")
        elif choice == '3':
            print("Starting dashboard...")
            import subprocess
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/dashboard/app.py'])
        else:
            print("Exiting.")

if __name__ == "__main__":
    main()
