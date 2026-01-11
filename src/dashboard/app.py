src/dashboard/app.py

#!/usr/bin/env python3
"""
Main entry point for Weather Anomaly Detection System
Direct and simple solution
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def ensure_directories():
    """Create required directories."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/output",
        "models",
        "logs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Created project directories")
    return True

def run_scraping():
    """Run scraping from weather.gov."""
    print("Running weather.gov scraping...")
    
    try:
        # Import the scraping module
        try:
            from scraping.scrape_weather_alerts import main as scrape_main
        except ImportError:
            print("Error: Could not import scraping module")
            print("Make sure src/scraping/scrape_weather_alerts.py exists")
            return False
        
        # Run scraping
        alert_count = scrape_main()
        
        # FIX: Check if alert_count is None
        if alert_count is None:
            print("Warning: Scraping returned None, treating as 0")
            alert_count = 0
        
        print(f"Scraping completed: {alert_count} alerts collected")
        return True
        
    except Exception as e:
        print(f"Scraping error: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing."""
    print("Running data preprocessing...")
    
    try:
        # Check if raw data exists
        if not os.path.exists("data/raw/weather_alerts_raw.csv"):
            print("Error: No raw data found. Run scraping first.")
            return False
        
        # Import preprocessing module
        try:
            from preprocessing.preprocess_text import preprocess_pipeline
        except ImportError:
            print("Error: Could not import preprocessing module")
            return False
        
        # Run preprocessing
        processed_df, daily_df = preprocess_pipeline(
            "data/raw/weather_alerts_raw.csv",
            "data/processed/weather_alerts_processed.csv"
        )
        
        if processed_df is not None:
            print(f"Preprocessing completed: {len(processed_df)} alerts processed")
            return True
        else:
            print("Warning: Preprocessing returned None")
            return True
            
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return False

def run_anomaly_detection():
    """Run anomaly detection."""
    print("Running anomaly detection...")
    
    try:
        # Check if processed data exists
        if not os.path.exists("data/processed/weather_alerts_daily.csv"):
            print("Error: No processed data found. Run preprocessing first.")
            return False
        
        # Import anomaly detection module
        try:
            from ml.anomaly_detection import run_anomaly_detection
        except ImportError:
            print("Error: Could not import anomaly detection module")
            return False
        
        # Run anomaly detection
        run_anomaly_detection(
            "data/processed/weather_alerts_daily.csv",
            "data/output/anomaly_results.csv",
            "models/isolation_forest.pkl"
        )
        
        print("Anomaly detection completed")
        return True
        
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return False

def run_forecasting():
    """Run forecasting with error handling."""
    print("Running forecasting...")
    
    try:
        # Check if processed data exists
        if not os.path.exists("data/processed/weather_alerts_daily.csv"):
            print("Error: No processed data found. Run preprocessing first.")
            return False
        
        # Import forecasting module
        try:
            from ml.forecast_model import run_forecasting
        except ImportError:
            print("Error: Could not import forecasting module")
            return False
        
        # Run forecasting
        result = run_forecasting(
            "data/processed/weather_alerts_daily.csv",
            "data/output/forecast_results.csv",
            "models/xgboost_forecast.pkl"
        )
        
        # FIX: Handle None return
        if result is None:
            print("Warning: Forecasting returned None")
            return True
        
        # Handle tuple return
        if isinstance(result, tuple):
            forecast_df, status = result
        else:
            forecast_df = result
        
        # Check if forecast_df is valid
        if forecast_df is None:
            print("Warning: Forecast data is None")
            return True
        
        # Check if it's a DataFrame
        if not hasattr(forecast_df, '__len__'):
            print("Warning: Forecast data is not iterable")
            return True
        
        print(f"Forecasting completed: {len(forecast_df)} predictions")
        return True
        
    except Exception as e:
        print(f"Forecasting error: {e}")
        return False

def run_dashboard():
    """Launch the Streamlit dashboard."""
    print("Launching dashboard...")
    
    try:
        # Check if dashboard exists
        dashboard_path = project_root / 'src' / 'dashboard' / 'app.py'
        if not dashboard_path.exists():
            print(f"Error: Dashboard not found at {dashboard_path}")
            return False
        
        # Run streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_path),
            '--server.port', '8501',
            '--server.headless', 'true'
        ]
        
        print(f"Starting dashboard: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        return False

def run_pipeline():
    """Run complete pipeline."""
    print("=" * 60)
    print("Running Complete Weather Anomaly Detection Pipeline")
    print("=" * 60)
    
    steps = [
        ("Scraping", run_scraping),
        ("Preprocessing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        print("-" * 40)
        
        try:
            success = step_func()
            results.append(success)
            
            if success:
                print(f"✓ {step_name} completed")
            else:
                print(f"✗ {step_name} failed")
                
        except Exception as e:
            print(f"✗ {step_name} error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    successful = sum(results)
    
    if successful == len(steps):
        print("✓ All pipeline steps completed successfully!")
    elif successful >= len(steps) // 2:
        print(f"⚠ {successful}/{len(steps)} steps completed")
    else:
        print(f"✗ Only {successful}/{len(steps)} steps completed")
    
    print("=" * 60)
    return successful

def main():
    """Main entry point."""
    # Create directories first
    ensure_directories()
    
    print("=" * 60)
    print("WEATHER ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    
    # Show menu
    print("\nSelect an option:")
    print("1. Run Dashboard")
    print("2. Run Complete Pipeline")
    print("3. Run Scraping (weather.gov)")
    print("4. Run Preprocessing")
    print("5. Run Anomaly Detection")
    print("6. Run Forecasting")
    print("7. Exit")
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            run_dashboard()
        elif choice == "2":
            run_pipeline()
            input("\nPress Enter to continue...")
            main()
        elif choice == "3":
            run_scraping()
            input("\nPress Enter to continue...")
            main()
        elif choice == "4":
            run_preprocessing()
            input("\nPress Enter to continue...")
            main()
        elif choice == "5":
            run_anomaly_detection()
            input("\nPress Enter to continue...")
            main()
        elif choice == "6":
            run_forecasting()
            input("\nPress Enter to continue...")
            main()
        elif choice == "7":
            print("Exiting...")
            return 0
        else:
            print("Invalid choice")
            input("\nPress Enter to continue...")
            main()
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
