#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Auto Start
Automatically runs the pipeline and launches dashboard
"""

import os
import sys
import subprocess
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_pipeline():
    """Run complete backend pipeline."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting pipeline...")
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        from preprocessing.preprocess_text import preprocess_pipeline
        from ml.anomaly_detection import run_anomaly_detection
        from ml.forecast_model import run_forecasting
        
        # Step 1: Scraping
        print("Step 1: Scraping weather data...")
        alert_count = scrape_main()
        print(f"Collected {alert_count} alerts")
        
        # Step 2: Preprocessing
        print("Step 2: Processing data...")
        processed_df, daily_df = preprocess_pipeline(
            "data/raw/weather_alerts_raw.csv",
            "data/processed/weather_alerts_processed.csv"
        )
        print(f"Processed {len(processed_df)} alerts")
        
        # Step 3: Anomaly Detection
        print("Step 3: Detecting anomalies...")
        anomaly_df, explanations = run_anomaly_detection(
            "data/processed/weather_alerts_daily.csv",
            "data/output/anomaly_results.csv",
            "models/isolation_forest.pkl"
        )
        print(f"Anomaly detection completed")
        
        # Step 4: Forecasting
        print("Step 4: Generating forecasts...")
        forecast_df, status = run_forecasting(
            "data/processed/weather_alerts_daily.csv",
            "data/output/forecast_results.csv",
            "models/xgboost_forecast.pkl"
        )
        print(f"Forecasting completed")
        
        print("\nPipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard."""
    print("Launching dashboard...")
    
    # Find dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard', 'app.py')
    
    if not os.path.exists(dashboard_path):
        print(f"Dashboard not found: {dashboard_path}")
        return False
    
    # Launch Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.port", "8501"]
    
    try:
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False

def main():
    """Main function - runs everything automatically."""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("="*60)
    print("WEATHER ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    # Run pipeline
    success = run_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("Starting dashboard...")
        print("="*60)
        launch_dashboard()
    else:
        print("\nPipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()
