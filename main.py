#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Main Entry Point
Connects to all backend modules in src/
"""

import os
import sys
import argparse
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_scraping():
    """Run web scraping from weather.gov."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running scraping...")
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        count = scrape_main()
        print(f"Scraping completed. Alerts: {count}")
        return True
    except Exception as e:
        print(f"Scraping error: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running preprocessing...")
    
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        
        # Check if raw data exists
        if not os.path.exists("data/raw/weather_alerts_raw.csv"):
            print("Error: No raw data found. Run scraping first.")
            return False
        
        processed_df, daily_df = preprocess_pipeline(
            "data/raw/weather_alerts_raw.csv",
            "data/processed/weather_alerts_processed.csv"
        )
        
        if processed_df is not None:
            print(f"Preprocessing completed. Alerts: {len(processed_df)}")
            return True
        return False
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return False

def run_anomaly_detection():
    """Run anomaly detection."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running anomaly detection...")
    
    try:
        from ml.anomaly_detection import run_anomaly_detection
        
        if not os.path.exists("data/processed/weather_alerts_daily.csv"):
            print("Error: No processed data found.")
            return False
        
        result_df, explanations = run_anomaly_detection(
            "data/processed/weather_alerts_daily.csv",
            "data/output/anomaly_results.csv",
            "models/isolation_forest.pkl"
        )
        
        print(f"Anomaly detection completed. Results: {len(result_df)}")
        return True
        
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return False

def run_forecasting():
    """Run forecasting."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running forecasting...")
    
    try:
        from ml.forecast_model import run_forecasting
        
        if not os.path.exists("data/processed/weather_alerts_daily.csv"):
            print("Error: No processed data found.")
            return False
        
        forecast_df, status = run_forecasting(
            "data/processed/weather_alerts_daily.csv",
            "data/output/forecast_results.csv",
            "models/xgboost_forecast.pkl"
        )
        
        print(f"Forecasting completed. Forecasts: {len(forecast_df)}")
        return True
        
    except Exception as e:
        print(f"Forecasting error: {e}")
        return False

def run_complete_pipeline():
    """Run complete end-to-end pipeline."""
    print("\n" + "="*60)
    print("WEATHER ANOMALY DETECTION PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    steps = [
        ("Scraping", run_scraping),
        ("Preprocessing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        success = step_func()
        results.append(success)
    
    # Summary
    print("\n" + "="*60)
    successful = sum(results)
    total = len(steps)
    
    print(f"PIPELINE COMPLETED: {successful}/{total} steps successful")
    
    if successful == total:
        print("Status: All steps completed successfully")
    elif successful >= total // 2:
        print("Status: Partial success - check logs for details")
    else:
        print("Status: Most steps failed - system needs attention")
    
    print("="*60)
    return successful

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='Weather Anomaly Detection System Backend'
    )
    
    parser.add_argument(
        'command',
        choices=['all', 'scrape', 'preprocess', 'anomaly', 'forecast'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'all':
        run_complete_pipeline()
    elif args.command == 'scrape':
        run_scraping()
    elif args.command == 'preprocess':
        run_preprocessing()
    elif args.command == 'anomaly':
        run_anomaly_detection()
    elif args.command == 'forecast':
        run_forecasting()

if __name__ == "__main__":
    main()
