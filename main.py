#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Main Entry Point
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def setup_directories():
    """Create required project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/output",
        "models",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Project directories created/verified")
    return True

def run_scraping():
    """Run data collection from weather sources."""
    print("Running data collection...")
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        result = scrape_main()
        
        if result is None:
            print("Warning: Data collection returned None")
            return False
        
        print(f"Data collection completed: {result} alerts collected")
        return True
        
    except ImportError as e:
        print(f"Error: Could not import scraping module: {e}")
        return False
    except Exception as e:
        print(f"Data collection error: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing pipeline."""
    print("Running data preprocessing...")
    
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        
        input_path = "data/raw/weather_alerts_raw.csv"
        output_path = "data/processed/weather_alerts_processed.csv"
        
        if not os.path.exists(input_path):
            print("Error: No raw data found. Run data collection first.")
            return False
        
        result = preprocess_pipeline(input_path, output_path)
        
        if result is None:
            print("Warning: Preprocessing returned None")
            return False
        
        print("Data preprocessing completed")
        return True
        
    except ImportError as e:
        print(f"Error: Could not import preprocessing module: {e}")
        return False
    except Exception as e:
        print(f"Data preprocessing error: {e}")
        return False

def run_anomaly_detection():
    """Run anomaly detection analysis."""
    print("Running anomaly detection...")
    
    try:
        from ml.anomaly_detection import run_anomaly_detection
        
        input_path = "data/processed/weather_alerts_daily.csv"
        output_path = "data/output/anomaly_results.csv"
        
        if not os.path.exists(input_path):
            print("Error: No processed data found. Run preprocessing first.")
            return False
        
        result = run_anomaly_detection(input_path, output_path)
        
        if result is None:
            print("Warning: Anomaly detection returned None")
            return False
        
        print("Anomaly detection completed")
        return True
        
    except ImportError as e:
        print(f"Error: Could not import anomaly detection module: {e}")
        return False
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return False

def run_forecasting():
    """Run forecasting analysis."""
    print("Running forecasting...")
    
    try:
        from ml.forecast_model import run_forecasting
        
        input_path = "data/processed/weather_alerts_daily.csv"
        output_path = "data/output/forecast_results.csv"
        
        if not os.path.exists(input_path):
            print("Error: No processed data found. Run preprocessing first.")
            return False
        
        result = run_forecasting(input_path, output_path)
        
        if result is None:
            print("Warning: Forecasting returned None")
            return False
        
        print("Forecasting completed")
        return True
        
    except ImportError as e:
        print(f"Error: Could not import forecasting module: {e}")
        return False
    except Exception as e:
        print(f"Forecasting error: {e}")
        return False

def run_dashboard():
    """Launch the monitoring dashboard."""
    print("Launching dashboard...")
    
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501"
        ])
        return True
    except Exception as e:
        print(f"Dashboard error: {e}")
        return False

def run_complete_pipeline():
    """Run the complete analysis pipeline."""
    print("=" * 50)
    print("Starting complete analysis pipeline")
    print("=" * 50)
    
    steps = [
        ("Data Collection", run_scraping),
        ("Data Preprocessing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    
    for step_name, step_function in steps:
        print(f"\n{step_name}:")
        print("-" * 30)
        
        try:
            success = step_function()
            results.append(success)
            
            if success:
                print(f"Completed: {step_name}")
            else:
                print(f"Failed: {step_name}")
                
        except Exception as e:
            print(f"Error in {step_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    successful = sum(results)
    
    if successful == len(steps):
        print("All pipeline steps completed successfully")
    elif successful >= len(steps) // 2:
        print(f"{successful}/{len(steps)} steps completed")
    else:
        print(f"Only {successful}/{len(steps)} steps completed")
    
    print("=" * 50)
    return successful

def main():
    """Main entry point."""
    # Setup project structure
    setup_directories()
    
    print("\n" + "=" * 50)
    print("Weather Anomaly Detection System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Launch Dashboard")
        print("2. Run Complete Analysis Pipeline")
        print("3. Run Data Collection")
        print("4. Run Data Preprocessing")
        print("5. Run Anomaly Detection")
        print("6. Run Forecasting")
        print("7. Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                run_dashboard()
            elif choice == "2":
                run_complete_pipeline()
            elif choice == "3":
                run_scraping()
            elif choice == "4":
                run_preprocessing()
            elif choice == "5":
                run_anomaly_detection()
            elif choice == "6":
                run_forecasting()
            elif choice == "7":
                print("Exiting system")
                sys.exit(0)
            else:
                print("Invalid selection")
                
        except KeyboardInterrupt:
            print("\nExiting system")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
