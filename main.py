#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Main Entry Point
"""

import os
import sys
import subprocess
from pathlib import Path
import time

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
    
    print("✓ Project directories ready")
    return True

def run_scraping():
    """Run REAL data collection from weather.gov."""
    print("\n🔍 Collecting REAL weather data from NOAA/NWS...")
    
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        
        print("Connecting to weather.gov...")
        result = scrape_main()
        
        if result is None:
            result = 0
        
        result = int(result)
        
        if result > 0:
            print(f"✓ Successfully collected {result} REAL weather alerts")
        elif result == 0:
            print("✓ System checked for alerts. No active alerts found.")
        else:
            print("⚠ Data collection completed with issues")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error: Could not import scraping module: {e}")
        return False
    except Exception as e:
        print(f"✗ Data collection error: {e}")
        return False

def run_preprocessing():
    """Process the collected data."""
    print("\n📊 Processing weather data...")
    
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        
        input_path = "data/raw/weather_alerts_raw.csv"
        
        if not os.path.exists(input_path):
            print("✗ No data found. Run data collection first.")
            return False
        
        print("Processing alerts...")
        processed_df, daily_df = preprocess_pipeline(
            input_path,
            "data/processed/weather_alerts_processed.csv"
        )
        
        if processed_df is not None:
            print(f"✓ Processed {len(processed_df)} alerts")
            if daily_df is not None:
                print(f"✓ Created {len(daily_df)} days of statistics")
            return True
        else:
            print("⚠ Processing may have had issues")
            return True
            
    except ImportError as e:
        print(f"✗ Error: Could not import preprocessing module: {e}")
        return False
    except Exception as e:
        print(f"✗ Processing error: {e}")
        return False

def run_anomaly_detection():
    """Detect anomalies in weather patterns."""
    print("\n🔬 Analyzing for anomalies...")
    
    try:
        from ml.anomaly_detection import run_anomaly_detection
        
        input_path = "data/processed/weather_alerts_daily.csv"
        
        if not os.path.exists(input_path):
            print("✗ No processed data found. Run preprocessing first.")
            return False
        
        print("Running anomaly detection...")
        result_df, explanations = run_anomaly_detection(
            input_path,
            "data/output/anomaly_results.csv"
        )
        
        if 'is_anomaly' in result_df.columns:
            anomaly_count = result_df['is_anomaly'].sum()
            print(f"✓ Found {anomaly_count} potential anomalies")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error: Could not import anomaly detection module: {e}")
        return False
    except Exception as e:
        print(f"✗ Anomaly detection error: {e}")
        return False

def run_forecasting():
    """Generate weather forecasts."""
    print("\n📈 Generating forecasts...")
    
    try:
        from ml.forecast_model import run_forecasting
        
        input_path = "data/processed/weather_alerts_daily.csv"
        
        if not os.path.exists(input_path):
            print("✗ No processed data found. Run preprocessing first.")
            return False
        
        print("Running forecasting...")
        forecast_df, status = run_forecasting(
            input_path,
            "data/output/forecast_results.csv"
        )
        
        if forecast_df is not None and not forecast_df.empty:
            print(f"✓ Generated {len(forecast_df)} forecast predictions")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error: Could not import forecasting module: {e}")
        return False
    except Exception as e:
        print(f"✗ Forecasting error: {e}")
        return False

def run_dashboard():
    """Launch the dashboard to view results."""
    print("\n📊 Launching dashboard...")
    
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"✗ Dashboard not found at {dashboard_path}")
        return False
    
    try:
        print("Starting Streamlit dashboard...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ])
        return True
    except Exception as e:
        print(f"✗ Dashboard error: {e}")
        return False

def run_complete_pipeline():
    """Run the complete analysis pipeline with REAL data."""
    print("\n" + "="*60)
    print("STARTING COMPLETE ANALYSIS PIPELINE")
    print("Collecting and analyzing REAL weather data")
    print("="*60)
    
    steps = [
        ("Data Collection", run_scraping),
        ("Data Processing", run_preprocessing),
        ("Anomaly Detection", run_anomaly_detection),
        ("Forecasting", run_forecasting)
    ]
    
    results = []
    
    for step_name, step_function in steps:
        print(f"\n[{len(results)+1}/{len(steps)}] {step_name}")
        print("-" * 40)
        
        try:
            success = step_function()
            results.append(success)
            
            if success:
                print(f"✓ {step_name} completed")
            else:
                print(f"✗ {step_name} failed")
                
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            results.append(False)
        
        time.sleep(1)  # Brief pause between steps
    
    # Summary
    print("\n" + "="*60)
    successful = sum(results)
    
    if successful == len(steps):
        print("🎉 COMPLETE PIPELINE SUCCESS!")
        print("All steps completed successfully")
    elif successful >= len(steps) // 2:
        print(f"⚠ PARTIAL SUCCESS: {successful}/{len(steps)} steps completed")
    else:
        print(f"❌ PIPELINE ISSUES: Only {successful}/{len(steps)} steps completed")
    
    print("="*60)
    return successful

def main():
    """Main entry point."""
    setup_directories()
    
    print("\n" + "="*60)
    print("WEATHER ANOMALY DETECTION SYSTEM")
    print("="*60)
    print("Using REAL data from NOAA/NWS weather.gov")
    print("="*60)
    
    while True:
        print("\nSelect an option:")
        print("1. Launch Dashboard (view results)")
        print("2. Run Complete Pipeline (collect & analyze REAL data)")
        print("3. Collect Weather Data Only")
        print("4. Process Collected Data")
        print("5. Detect Anomalies")
        print("6. Generate Forecasts")
        print("7. Exit")
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == "1":
                run_dashboard()
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
                print("Exiting system...")
                sys.exit(0)
            else:
                print("Invalid selection")
                
        except KeyboardInterrupt:
            print("\n\nExiting system...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
