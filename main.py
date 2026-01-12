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
        
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'scraping_interval_hours': 1,
            'processing_interval_hours': 1,
            'ml_interval_hours': 6,
            'max_days_to_keep': 90,
            'anomaly_contamination': 0.1,
            'forecast_horizon_days': 7,
            'enable_dashboard': True,
            'dashboard_port': 8501,
            'log_level': 'INFO'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        return default_config
    
    def run_scraping(self):
        """Run the web scraping process."""
        logger.info("Starting web scraping process...")
        try:
            scrape_main()
            logger.info("Web scraping completed successfully")
            return True
        except Exception as e:
            logger.error(f"Web scraping failed: {str(e)}")
            return False
    
    def run_preprocessing(self):
        """Run the data preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        try:
            preprocess_pipeline(self.raw_data_path, self.processed_data_path)
            logger.info("Data preprocessing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            return False
    
    def run_anomaly_detection_pipeline(self):
        """Run the anomaly detection pipeline."""
        logger.info("Starting anomaly detection...")
        try:
            if not os.path.exists(self.daily_data_path):
                logger.warning("Daily data not found. Running preprocessing first.")
                if not self.run_preprocessing():
                    return False
            
            anomaly_results, explanations = run_anomaly_detection(
                self.daily_data_path,
                self.anomaly_output_path,
                self.anomaly_model_path
            )
            
            logger.info(f"Anomaly detection completed. Found {anomaly_results['is_anomaly'].sum()} anomalies")
            return True
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return False
    
    def run_forecasting_pipeline(self):
        """Run the forecasting pipeline."""
        logger.info("Starting forecasting...")
        try:
            if not os.path.exists(self.daily_data_path):
                logger.warning("Daily data not found. Running preprocessing first.")
                if not self.run_preprocessing():
                    return False
            
            forecast_results, evaluation = run_forecasting(
                self.daily_data_path,
                self.forecast_output_path,
                self.forecast_model_path
            )
            
            logger.info(f"Forecasting completed for {len(forecast_results['target'].unique())} targets")
            return True
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            return False
    
    def generate_insights(self):
        """Generate plain English insights from analysis results."""
        logger.info("Generating insights...")
        try:
            # Load data
            daily_stats = pd.read_csv(self.daily_data_path) if os.path.exists(self.daily_data_path) else pd.DataFrame()
            anomalies = pd.read_csv(self.anomaly_output_path) if os.path.exists(self.anomaly_output_path) else pd.DataFrame()
            forecasts = pd.read_csv(self.forecast_output_path) if os.path.exists(self.forecast_output_path) else pd.DataFrame()
            
            # Parse dates
            if not daily_stats.empty and 'issued_date' in daily_stats.columns:
                daily_stats['issued_date'] = pd.to_datetime(daily_stats['issued_date'])
            
            if not anomalies.empty and 'issued_date' in anomalies.columns:
                anomalies['issued_date'] = pd.to_datetime(anomalies['issued_date'])
                anomalies = anomalies.set_index('issued_date')
            
            # Generate insights
            insights = generate_plain_english_insights(daily_stats, anomalies, forecasts)
            
            # Save insights
            insights_data = {
                'generated_at': datetime.now().isoformat(),
                'insights': insights,
                'summary': f"Generated {len(insights)} insights"
            }
            
            insights_path = "data/output/insights.json"
            save_to_json(insights_data, insights_path)
            
            logger.info(f"Generated {len(insights)} insights")
            return True
        except Exception as e:
            logger.error(f"Insights generation failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete end-to-end pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE PIPELINE RUN")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {
            'scraping': False,
            'preprocessing': False,
            'anomaly_detection': False,
            'forecasting': False,
            'insights': False
        }
        
        try:
            # Step 1: Scraping
            results['scraping'] = self.run_scraping()
            
            if results['scraping']:
                # Step 2: Preprocessing
                results['preprocessing'] = self.run_preprocessing()
                
                if results['preprocessing']:
                    # Step 3: Anomaly Detection
                    results['anomaly_detection'] = self.run_anomaly_detection_pipeline()
                    
                    # Step 4: Forecasting
                    results['forecasting'] = self.run_forecasting_pipeline()
                    
                    # Step 5: Insights
                    results['insights'] = self.generate_insights()
            
            # Log summary
            elapsed_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info("PIPELINE RUN COMPLETE")
            logger.info("=" * 60)
            
            for step, success in results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                logger.info(f"{step.upper():20} {status}")
            
            logger.info(f"Total time: {elapsed_time:.2f} seconds")
            logger.info("=" * 60)
            
            return all(results.values())
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {str(e)}")
            return False
    
    def cleanup_old_data(self):
        """Clean up old data files."""
        logger.info("Cleaning up old data...")
        try:
            from utils.helpers import cleanup_old_files
            
            # Clean up old raw files
            cleanup_old_files('data/raw', days_to_keep=self.config['max_days_to_keep'])
            
            # Clean up old logs
            cleanup_old_files('logs', days_to_keep=30, pattern='*.log')
            
            logger.info("Cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return False
    
    def schedule_jobs(self):
        """Schedule all periodic jobs."""
        # Scraping job (hourly)
        schedule.every(self.config['scraping_interval_hours']).hours.do(
            lambda: self.run_scraping()
        )
        
        # Preprocessing job (hourly, 5 minutes after scraping)
        schedule.every(self.config['scraping_interval_hours']).hours.at(":05").do(
            lambda: self.run_preprocessing()
        )
        
        # ML jobs (every 6 hours)
        schedule.every(self.config['ml_interval_hours']).hours.do(
            lambda: self.run_anomaly_detection_pipeline()
        )
        
        schedule.every(self.config['ml_interval_hours']).hours.at(":30").do(
            lambda: self.run_forecasting_pipeline()
        )
        
        # Insights generation (every 6 hours, after ML)
        schedule.every(self.config['ml_interval_hours']).hours.at(":45").do(
            lambda: self.generate_insights()
        )
        
        # Cleanup job (daily at 2 AM)
        schedule.every().day.at("02:00").do(
            lambda: self.cleanup_old_data()
        )
        
        logger.info(f"Scheduled jobs:")
        logger.info(f"  - Scraping: every {self.config['scraping_interval_hours']} hours")
        logger.info(f"  - Preprocessing: every {self.config['processing_interval_hours']} hours")
        logger.info(f"  - ML analysis: every {self.config['ml_interval_hours']} hours")
        logger.info(f"  - Cleanup: daily at 02:00")
    
    def run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("Starting scheduler...")
        self.running = True
        
        # Schedule jobs
        self.schedule_jobs()
        
        # Run initial pipeline
        logger.info("Running initial pipeline...")
        self.run_complete_pipeline()
        
        # Start scheduler loop
        logger.info("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
        finally:
            self.running = False
    
    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        if not self.config['enable_dashboard']:
            logger.info("Dashboard disabled in configuration")
            return
        
        logger.info(f"Starting dashboard on port {self.config['dashboard_port']}...")
        
        # In production, you would run this in a separate process
        # For now, we'll just provide instructions
        logger.info("To start the dashboard, run: streamlit run src/dashboard/app.py")
        
        # Alternatively, you can use subprocess to start it
        # import subprocess
        # subprocess.Popen([
        #     'streamlit', 'run', 'src/dashboard/app.py',
        #     '--server.port', str(self.config['dashboard_port'])
        # ])

def main():
    """Main entry point for the system."""
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

if __name__ == "__main__":
    # Import pandas here to avoid early import issues
    import pandas as pd
    
    main()
