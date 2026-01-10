#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Main Entry Point
ONLY pipeline execution - no dashboard code
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def setup_directories():
    """Create required directories."""
    dirs = ["data/raw", "data/processed", "data/output", "models", "logs"]
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)
    return True

def run_scraping():
    """Run data collection."""
    print("=" * 50)
    print("DATA COLLECTION")
    print("=" * 50)
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        result = scrape_main()
        
        if result is None:
            result = 0
        result = int(result)
        
        if result > 0:
            print(f"✓ Collected {result} alerts")
        else:
            print("✓ No alerts found (or 0 alerts)")
        
        # Check if file was created
        raw_file = project_root / "data" / "raw" / "weather_alerts_raw.csv"
        if raw_file.exists():
            print(f"✓ Data saved to: {raw_file}")
        else:
            print("⚠ Data file not created")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing."""
    print("=" * 50)
    print("DATA PROCESSING")
    print("=" * 50)
    
    try:
        # Check for input
        raw_file = project_root / "data" / "raw" / "weather_alerts_raw.csv"
        if not raw_file.exists():
            print("✗ No data found. Run scraping first.")
            return False
        
        from preprocessing.preprocess_text import preprocess_pipeline
        
        result = preprocess_pipeline(
            str(raw_file),
            str(project_root / "data" / "processed" / "weather_alerts_processed.csv")
        )
        
        if result:
            print("✓ Processing completed")
            daily_file = project_root / "data" / "processed" / "weather_alerts_daily.csv"
            if daily_file.exists():
                print(f"✓ Daily stats: {daily_file}")
            return True
        else:
            print("⚠ Processing had issues")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_anomaly_detection():
    """Run anomaly detection."""
    print("=" * 50)
    print("ANOMALY DETECTION")
    print("=" * 50)
    
    try:
        daily_file = project_root / "data" / "processed" / "weather_alerts_daily.csv"
        if not daily_file.exists():
            print("✗ No processed data. Run preprocessing first.")
            return False
        
        from ml.anomaly_detection import run_anomaly_detection as run_anomaly
        
        result = run_anomaly(
            str(daily_file),
            str(project_root / "data" / "output" / "anomaly_results.csv"),
            str(project_root / "models" / "isolation_forest.pkl")
        )
        
        if result:
            print("✓ Anomaly detection completed")
            return True
        else:
            print("⚠ Anomaly detection had issues")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_forecasting():
    """Run forecasting."""
    print("=" * 50)
    print("FORECASTING")
    print("=" * 50)
    
    try:
        daily_file = project_root / "data" / "processed" / "weather_alerts_daily.csv"
        if not daily_file.exists():
            print("✗ No processed data. Run preprocessing first.")
            return False
        
        from ml.forecast_model import run_forecasting as run_forecast
        
        result = run_forecast(
            str(daily_file),
            str(project_root / "data" / "output" / "forecast_results.csv"),
            str(project_root / "models" / "xgboost_forecast.pkl")
        )
        
        if result:
            print("✓ Forecasting completed")
            return True
        else:
            print("⚠ Forecasting had issues")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def launch_dashboard():
    """Launch the dashboard."""
    print("=" * 50)
    print("LAUNCHING DASHBOARD")
    print("=" * 50)
    
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"✗ Dashboard not found: {dashboard_path}")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501"
        ])
        return True
    except Exception as e:
        print(f"✗ Dashboard error: {e}")
        return False

def run_complete_pipeline():
    """Run all steps."""
    print("=" * 60)
    print("COMPLETE PIPELINE")
    print("=" * 60)
    
    steps = [
        ("Data Collection", run_scraping),
        ("Data Processing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    for name, func in steps:
        print(f"\n{name}:")
        print("-" * 30)
        success = func()
        results.append(success)
        time.sleep(1)
    
    print("\n" + "=" * 60)
    successful = sum(results)
    print(f"Results: {successful}/{len(steps)} steps successful")
    print("=" * 60)
    
    return successful >= len(steps) // 2

def main():
    """Main menu."""
    setup_directories()
    
    print("\n" + "=" * 60)
    print("WEATHER ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Launch Dashboard")
        print("2. Run Complete Pipeline")
        print("3. Run Data Collection")
        print("4. Run Data Processing")
        print("5. Run Anomaly Detection")
        print("6. Run Forecasting")
        print("7. Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                launch_dashboard()
            elif choice == "2":
                run_complete_pipeline()
                input("\nPress Enter to continue...")
            elif choice == "3":
                run_scraping()
                input("\nPress Enter to continue...")
            elif choice == "4":
                run_preprocessing()
                input("\nPress Enter to continue...")
            elif choice == "5":
                run_anomaly_detection()
                input("\nPress Enter to continue...")
            elif choice == "6":
                run_forecasting()
                input("\nPress Enter to continue...")
            elif choice == "7":
                print("Exiting...")
                sys.exit(0)
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
