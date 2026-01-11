#!/usr/bin/env python3
"""
Complete pipeline script - runs all steps
"""
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def run_scraping():
    """Run data scraping."""
    print("\n" + "="*50)
    print("STEP 1: Data Collection")
    print("="*50)
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        result = scrape_main()
        print(f"✓ Scraping completed: {result} alerts")
        return True
    except Exception as e:
        print(f"✗ Scraping error: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing."""
    print("\n" + "="*50)
    print("STEP 2: Data Processing")
    print("="*50)
    
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        result = preprocess_pipeline(
            "data/raw/weather_alerts_raw.csv",
            "data/processed/weather_alerts_processed.csv"
        )
        print("✓ Preprocessing completed")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def run_anomaly_detection():
    """Run anomaly detection."""
    print("\n" + "="*50)
    print("STEP 3: Anomaly Detection")
    print("="*50)
    
    try:
        from ml.anomaly_detection import run_anomaly_detection
        result = run_anomaly_detection(
            "data/processed/weather_alerts_daily.csv",
            "data/output/anomaly_results.csv",
            "models/isolation_forest.pkl"
        )
        print("✓ Anomaly detection completed")
        return True
    except Exception as e:
        print(f"✗ Anomaly detection error: {e}")
        return False

def run_forecasting():
    """Run forecasting."""
    print("\n" + "="*50)
    print("STEP 4: Forecasting")
    print("="*50)
    
    try:
        from ml.forecast_model import run_forecasting
        result = run_forecasting(
            "data/processed/weather_alerts_daily.csv",
            "data/output/forecast_results.csv",
            "models/xgboost_forecast.pkl"
        )
        print("✓ Forecasting completed")
        return True
    except Exception as e:
        print(f"✗ Forecasting error: {e}")
        return False

def main():
    """Run complete pipeline."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE WEATHER ANALYSIS PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    steps = [
        ("Data Collection", run_scraping),
        ("Data Processing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    for name, func in steps:
        print(f"\n{name}")
        print("-" * 40)
        success = func()
        results.append(success)
    
    # Summary
    print("\n" + "="*60)
    successful = sum(results)
    print(f"Pipeline completed: {successful}/{len(steps)} steps successful")
    print("="*60)
    
    if successful == len(steps):
        print("\n✓ All steps completed successfully!")
        print("\nNext: Launch dashboard from main.py (option 1)")
    else:
        print(f"\n⚠ {len(steps)-successful} steps failed")

if __name__ == "__main__":
    main()
