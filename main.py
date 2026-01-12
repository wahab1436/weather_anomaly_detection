#!/usr/bin/env python3
"""
Main orchestration script for Weather Anomaly Detection.
Runs the complete pipeline: scraping → preprocessing → daily stats → ML → dashboard.
"""

import os
import sys
import time
import logging
from datetime import datetime
import json
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import your modules
from scraping.scrape_weather_alerts import main as scrape_main
from preprocessing.preprocess_text import preprocess_pipeline
from ml.anomaly_detection import run_anomaly_detection
from ml.forecast_model import run_forecasting
from utils.helpers import setup_logging, generate_plain_english_insights, save_to_json, cleanup_old_files

# Setup logging
setup_logging('logs/system.log')
logger = logging.getLogger(__name__)

class WeatherAnomalySystem:
    """Main orchestration class for weather anomaly detection."""

    def __init__(self, config_path: str = None):
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

    def load_config(self, config_path: str = None):
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
                logger.error(f"Error loading config: {e}")
        return default_config

    def run_scraping(self):
        logger.info("Starting scraping...")
        try:
            scrape_main()
            logger.info("Scraping completed.")
            return True
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return False

    def run_preprocessing(self):
        logger.info("Preprocessing data...")
        try:
            preprocess_pipeline(self.raw_data_path, self.processed_data_path)
            logger.info("Preprocessing completed.")
            return True
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return False

    def generate_daily_stats(self):
        """Aggregate processed data into daily stats for anomaly detection and forecasting."""
        logger.info("Generating daily stats...")
        try:
            if not os.path.exists(self.processed_data_path):
                logger.warning("Processed file not found. Cannot generate daily stats.")
                return False

            df = pd.read_csv(self.processed_data_path)
            df['issued_date'] = pd.to_datetime(df['issued_date'])
            daily_stats = df.groupby(df['issued_date'].dt.date).agg({
                'type': 'count',
                'severity_score': 'mean'
            }).rename(columns={'type': 'total_alerts'})
            os.makedirs(os.path.dirname(self.daily_data_path), exist_ok=True)
            daily_stats.to_csv(self.daily_data_path)
            logger.info(f"Daily stats saved to {self.daily_data_path}")
            return True
        except Exception as e:
            logger.error(f"Generating daily stats failed: {e}")
            return False

    def run_anomaly_detection_pipeline(self):
        logger.info("Running anomaly detection...")
        if not os.path.exists(self.daily_data_path):
            logger.info("Daily stats not found. Generating now...")
            if not self.generate_daily_stats():
                return False
        try:
            anomaly_results, _ = run_anomaly_detection(
                self.daily_data_path,
                self.anomaly_output_path,
                self.anomaly_model_path
            )
            logger.info(f"Anomaly detection complete. Found {anomaly_results['is_anomaly'].sum()} anomalies")
            return True
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False

    def run_forecasting_pipeline(self):
        logger.info("Running forecasting...")
        if not os.path.exists(self.daily_data_path):
            logger.info("Daily stats not found. Generating now...")
            if not self.generate_daily_stats():
                return False
        try:
            forecast_results, _ = run_forecasting(
                self.daily_data_path,
                self.forecast_output_path,
                self.forecast_model_path
            )
            logger.info(f"Forecasting completed for {len(forecast_results)} entries")
            return True
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return False

    def generate_insights(self):
        logger.info("Generating insights...")
        try:
            daily_stats = pd.read_csv(self.daily_data_path) if os.path.exists(self.daily_data_path) else pd.DataFrame()
            anomalies = pd.read_csv(self.anomaly_output_path) if os.path.exists(self.anomaly_output_path) else pd.DataFrame()
            forecasts = pd.read_csv(self.forecast_output_path) if os.path.exists(self.forecast_output_path) else pd.DataFrame()
            insights = generate_plain_english_insights(daily_stats, anomalies, forecasts)
            save_to_json({'generated_at': datetime.now().isoformat(), 'insights': insights}, "data/output/insights.json")
            logger.info(f"Generated {len(insights)} insights")
            return True
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return False

    def cleanup_old_data(self):
        logger.info("Cleaning up old data...")
        try:
            cleanup_old_files('data/raw', days_to_keep=self.config['max_days_to_keep'])
            cleanup_old_files('logs', days_to_keep=30, pattern='*.log')
            logger.info("Cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def run_complete_pipeline(self):
        logger.info("="*50)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*50)

        results = {
            'scraping': False,
            'preprocessing': False,
            'daily_stats': False,
            'anomaly_detection': False,
            'forecasting': False,
            'insights': False
        }

        results['scraping'] = self.run_scraping()
        if results['scraping']:
            results['preprocessing'] = self.run_preprocessing()
            if results['preprocessing']:
                results['daily_stats'] = self.generate_daily_stats()
                if results['daily_stats']:
                    results['anomaly_detection'] = self.run_anomaly_detection_pipeline()
                    results['forecasting'] = self.run_forecasting_pipeline()
                    results['insights'] = self.generate_insights()

        # Log summary
        for step, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{step.upper():20} {status}")

        return all(results.values())

def main():
    parser = argparse.ArgumentParser(description="Weather Anomaly Detection System")
    parser.add_argument(
        'command',
        choices=['run', 'pipeline', 'scrape', 'preprocess', 'detect-anomalies', 'forecast', 'insights', 'cleanup'],
        help='Command to execute'
    )
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()

    system = WeatherAnomalySystem(args.config)

    if args.command in ['run', 'pipeline']:
        system.run_complete_pipeline()
    elif args.command == 'scrape':
        system.run_scraping()
    elif args.command == 'preprocess':
        system.run_preprocessing()
    elif args.command == 'detect-anomalies':
        system.run_anomaly_detection_pipeline()
    elif args.command == 'forecast':
        system.run_forecasting_pipeline()
    elif args.command == 'insights':
        system.generate_insights()
    elif args.command == 'cleanup':
        system.cleanup_old_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
