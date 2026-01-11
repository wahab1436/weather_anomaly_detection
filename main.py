#!/usr/bin/env python3
"""
WEATHER ANOMALY DETECTION SYSTEM - PRODUCTION ENTRY POINT
Complete backend pipeline connector
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Add src to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ============================================================================
# DIRECTORY MANAGEMENT - FIRST THING!
# ============================================================================

def ensure_directories():
    """Create all required directories - MUST BE CALLED FIRST!"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/output",
        "models",
        "logs",
        "backups"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return True

# CREATE DIRECTORIES IMMEDIATELY
ensure_directories()

# ============================================================================
# LOGGING SETUP - AFTER DIRECTORIES CREATED
# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Weather Anomaly Detection System initialized")

# ============================================================================
# BACKEND PIPELINE FUNCTIONS
# ============================================================================

def run_scraping() -> bool:
    """Run weather.gov scraping pipeline."""
    logger.info("STEP 1: Starting weather.gov data collection")
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        
        alert_count = scrape_main()
        
        if alert_count is None:
            logger.warning("Scraping returned None, checking for data file")
            if os.path.exists("data/raw/weather_alerts_raw.csv"):
                import pandas as pd
                df = pd.read_csv("data/raw/weather_alerts_raw.csv")
                alert_count = len(df)
                logger.info(f"Found existing data: {alert_count} alerts")
            else:
                logger.error("No data collected and no existing data found")
                return False
        
        logger.info(f"Data collection completed: {alert_count} alerts")
        return True
        
    except Exception as e:
        logger.error(f"Scraping pipeline failed: {str(e)}", exc_info=True)
        return False

def run_preprocessing() -> bool:
    """Run data preprocessing pipeline."""
    logger.info("STEP 2: Starting data preprocessing")
    
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        
        # Check for input data
        input_file = "data/raw/weather_alerts_raw.csv"
        if not os.path.exists(input_file):
            logger.error(f"Raw data not found: {input_file}")
            return False
        
        output_file = "data/processed/weather_alerts_processed.csv"
        
        processed_df, daily_df = preprocess_pipeline(input_file, output_file)
        
        if processed_df is None or processed_df.empty:
            logger.warning("Preprocessing returned empty dataframe")
            return False
        
        logger.info(f"Preprocessing completed: {len(processed_df)} alerts, {len(daily_df)} daily records")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}", exc_info=True)
        return False

def run_anomaly_detection() -> bool:
    """Run anomaly detection pipeline."""
    logger.info("STEP 3: Starting anomaly detection")
    
    try:
        from ml.anomaly_detection import run_anomaly_detection
        
        input_file = "data/processed/weather_alerts_daily.csv"
        if not os.path.exists(input_file):
            logger.error(f"Processed data not found: {input_file}")
            return False
        
        output_file = "data/output/anomaly_results.csv"
        model_file = "models/isolation_forest.pkl"
        
        result_df, explanations = run_anomaly_detection(
            input_file, 
            output_file, 
            model_file
        )
        
        if result_df is None or result_df.empty:
            logger.warning("Anomaly detection returned empty results")
            return False
        
        # Count anomalies
        anomaly_count = 0
        if 'is_anomaly' in result_df.columns:
            anomaly_count = result_df['is_anomaly'].sum()
        
        logger.info(f"Anomaly detection completed: {anomaly_count} anomalies detected")
        return True
        
    except Exception as e:
        logger.error(f"Anomaly detection pipeline failed: {str(e)}", exc_info=True)
        return False

def run_forecasting() -> bool:
    """Run forecasting pipeline."""
    logger.info("STEP 4: Starting forecasting")
    
    try:
        from ml.forecast_model import run_forecasting
        
        input_file = "data/processed/weather_alerts_daily.csv"
        if not os.path.exists(input_file):
            logger.error(f"Processed data not found: {input_file}")
            return False
        
        output_file = "data/output/forecast_results.csv"
        model_file = "models/xgboost_forecast.pkl"
        
        forecast_df, status = run_forecasting(
            input_file, 
            output_file, 
            model_file
        )
        
        if forecast_df is None or forecast_df.empty:
            logger.warning("Forecasting returned empty results")
            return False
        
        logger.info(f"Forecasting completed: {len(forecast_df)} forecast records")
        return True
        
    except Exception as e:
        logger.error(f"Forecasting pipeline failed: {str(e)}", exc_info=True)
        return False

# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def run_complete_pipeline() -> bool:
    """Run complete end-to-end pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE WEATHER ANOMALY DETECTION PIPELINE")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Pipeline steps
    pipeline_steps = [
        ("Data Collection", run_scraping),
        ("Data Preprocessing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    
    for step_name, step_function in pipeline_steps:
        step_start = datetime.now()
        logger.info(f"\n{'='*40}")
        logger.info(f"STARTING: {step_name}")
        logger.info(f"{'='*40}")
        
        try:
            success = step_function()
            results.append(success)
            
            step_duration = datetime.now() - step_start
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"COMPLETED: {step_name} - {status} ({step_duration})")
            
        except Exception as e:
            logger.error(f"STEP FAILED: {step_name} - {str(e)}", exc_info=True)
            results.append(False)
    
    # Calculate statistics
    total_duration = datetime.now() - start_time
    successful_steps = sum(results)
    total_steps = len(pipeline_steps)
    
    # Final report
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total Duration: {total_duration}")
    logger.info(f"Steps Successful: {successful_steps}/{total_steps}")
    logger.info(f"Start Time: {start_time}")
    logger.info(f"End Time: {datetime.now()}")
    
    return successful_steps >= 2  # At least 50% success

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description='Weather Anomaly Detection System - Production Backend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py all          # Run complete pipeline
  python main.py scrape       # Run only data collection
  python main.py preprocess   # Run only preprocessing
  python main.py anomaly      # Run only anomaly detection
  python main.py forecast     # Run only forecasting
        """
    )
    
    parser.add_argument(
        'command',
        choices=['all', 'scrape', 'preprocess', 'anomaly', 'forecast'],
        help='Pipeline command to execute'
    )
    
    args = parser.parse_args()
    
    # Execute command
    command_map = {
        'all': run_complete_pipeline,
        'scrape': run_scraping,
        'preprocess': run_preprocessing,
        'anomaly': run_anomaly_detection,
        'forecast': run_forecasting
    }
    
    # Run the selected command
    success = command_map[args.command]()
    
    # Exit code
    sys.exit(0 if success else 1)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
