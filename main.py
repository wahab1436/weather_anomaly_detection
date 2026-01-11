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

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def ensure_directories():
    """Create all required directories."""
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
        logger.info(f"Ensured directory: {directory}")
    
    return True

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

def run_initial_data_collection() -> bool:
    """Run initial data collection script."""
    logger.info("Running initial data collection")
    
    try:
        from scripts.initial_data_collection import collect_historical_data
        
        data = collect_historical_data(days_back=7)
        
        if data is None or data.empty:
            logger.warning("Initial data collection returned empty")
            return False
        
        logger.info(f"Initial data collection completed: {len(data)} records")
        return True
        
    except Exception as e:
        logger.error(f"Initial data collection failed: {str(e)}")
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
    
    # Ensure directories
    ensure_directories()
    
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
    
    # Check system files
    logger.info("\nSYSTEM STATUS CHECK:")
    system_files = [
        ("Raw Data", "data/raw/weather_alerts_raw.csv"),
        ("Processed Data", "data/processed/weather_alerts_daily.csv"),
        ("Anomaly Results", "data/output/anomaly_results.csv"),
        ("Forecast Results", "data/output/forecast_results.csv")
    ]
    
    for file_name, file_path in system_files:
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                logger.info(f"✓ {file_name}: {len(df)} records")
            except:
                logger.info(f"✓ {file_name}: EXISTS")
        else:
            logger.warning(f"✗ {file_name}: MISSING")
    
    logger.info("=" * 70)
    
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
  python main.py init         # Run initial data collection
        """
    )
    
    parser.add_argument(
        'command',
        choices=['all', 'scrape', 'preprocess', 'anomaly', 'forecast', 'init', 'status'],
        help='Pipeline command to execute'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    command_map = {
        'all': run_complete_pipeline,
        'scrape': run_scraping,
        'preprocess': run_preprocessing,
        'anomaly': run_anomaly_detection,
        'forecast': run_forecasting,
        'init': run_initial_data_collection,
        'status': lambda: True  # Placeholder
    }
    
    if args.command == 'status':
        print_system_status()
        return
    
    # Run the selected command
    success = command_map[args.command]()
    
    # Exit code
    sys.exit(0 if success else 1)

def print_system_status():
    """Print system status."""
    print("\n" + "="*70)
    print("WEATHER ANOMALY DETECTION SYSTEM - STATUS")
    print("="*70)
    
    # Check directories
    print("\nDIRECTORIES:")
    dirs = ["data/raw", "data/processed", "data/output", "models", "logs"]
    for d in dirs:
        exists = os.path.exists(d)
        print(f"  {'✓' if exists else '✗'} {d}")
    
    # Check files
    print("\nDATA FILES:")
    files = [
        ("Raw Alerts", "data/raw/weather_alerts_raw.csv"),
        ("Daily Stats", "data/processed/weather_alerts_daily.csv"),
        ("Anomalies", "data/output/anomaly_results.csv"),
        ("Forecasts", "data/output/forecast_results.csv"),
        ("Anomaly Model", "models/isolation_forest.pkl"),
        ("Forecast Model", "models/xgboost_forecast.pkl")
    ]
    
    for name, path in files:
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_csv(path)
                size = os.path.getsize(path) / 1024  # KB
                print(f"  ✓ {name}: {len(df)} records ({size:.1f} KB)")
            except:
                print(f"  ✓ {name}: EXISTS")
        else:
            print(f"  ✗ {name}: MISSING")
    
    # Module check
    print("\nMODULES:")
    modules = [
        ("Scraping", "scraping.scrape_weather_alerts"),
        ("Preprocessing", "preprocessing.preprocess_text"),
        ("Anomaly Detection", "ml.anomaly_detection"),
        ("Forecasting", "ml.forecast_model")
    ]
    
    for name, module_path in modules:
        try:
            __import__(module_path)
            print(f"  ✓ {name}: AVAILABLE")
        except ImportError:
            print(f"  ✗ {name}: UNAVAILABLE")
    
    print("\n" + "="*70)
    print("Status check completed")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
