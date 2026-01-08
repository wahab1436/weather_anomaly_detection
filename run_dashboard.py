#!/usr/bin/env python3
"""
Main runner script for Weather Anomaly Detection Dashboard
"""
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.utils.helpers import create_directories, setup_logging
from src.scraping.scrape_weather_alerts import WeatherAlertScraper, schedule_scraping
from src.preprocessing.preprocess_text import run_preprocessing_pipeline
from src.ml.anomaly_detection import run_anomaly_detection
from src.ml.forecast_model import run_forecast_pipeline
from src.dashboard.app import main as run_dashboard

def setup_project():
    """Setup the project structure and dependencies"""
    print("=" * 60)
    print("Weather Anomaly Detection Dashboard - Setup")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Create logs directory if not exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    print("\n2. Setting up logging...")
    logger = setup_logging()
    
    print("\n3. Checking dependencies...")
    try:
        import pandas
        import streamlit
        import sklearn
        import xgboost
        import plotly
        print("✓ All major dependencies are installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    return True

def run_scraping():
    """Run the web scraping job"""
    print("\n" + "=" * 60)
    print("Running Web Scraping")
    print("=" * 60)
    
    scraper = WeatherAlertScraper()
    count = scraper.run_scraping_job()
    
    print(f"\nScraping completed. Collected {count} new alerts.")
    return count > 0

def run_preprocessing():
    """Run data preprocessing"""
    print("\n" + "=" * 60)
    print("Running Data Preprocessing")
    print("=" * 60)
    
    df_proc, df_agg = run_preprocessing_pipeline()
    
    if df_proc is not None and df_agg is not None:
        print(f"\nPreprocessing completed.")
        print(f"Processed data: {len(df_proc)} rows, {len(df_proc.columns)} columns")
        print(f"Aggregated data: {len(df_agg)} rows, {len(df_agg.columns)} columns")
        return True
    else:
        print("\nPreprocessing failed.")
        return False

def run_anomaly():
    """Run anomaly detection"""
    print("\n" + "=" * 60)
    print("Running Anomaly Detection")
    print("=" * 60)
    
    df_anomalies = run_anomaly_detection()
    
    if df_anomalies is not None:
        anomaly_count = df_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_anomalies.columns else 0
        print(f"\nAnomaly detection completed.")
        print(f"Detected {anomaly_count} anomalies")
        return True
    else:
        print("\nAnomaly detection failed.")
        return False

def run_forecast():
    """Run forecasting"""
    print("\n" + "=" * 60)
    print("Running Forecasting")
    print("=" * 60)
    
    forecasts, metrics = run_forecast_pipeline()
    
    if forecasts is not None and metrics is not None:
        print(f"\nForecasting completed.")
        print(f"Test MAE: {metrics.get('test_mae', 0):.2f}")
        print(f"Test RMSE: {metrics.get('test_rmse', 0):.2f}")
        print(f"Generated {len(forecasts)} days of forecasts")
        return True
    else:
        print("\nForecasting failed.")
        return False

def run_full_pipeline():
    """Run the complete data pipeline"""
    print("\n" + "=" * 60)
    print("Running Full Data Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    steps = [
        ("Web Scraping", run_scraping),
        ("Data Preprocessing", run_preprocessing),
        ("Anomaly Detection", run_anomaly),
        ("Forecasting", run_forecast)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        start_time = time.time()
        
        try:
            success = step_func()
            elapsed = time.time() - start_time
            
            if success:
                print(f"✓ {step_name} completed in {elapsed:.1f} seconds")
                results[step_name] = {'success': True, 'time': elapsed}
            else:
                print(f"✗ {step_name} failed after {elapsed:.1f} seconds")
                results[step_name] = {'success': False, 'time': elapsed}
                
                # Ask if we should continue
                if step_name != "Web Scraping":  # Scraping might fail if no internet
                    continue_choice = input(f"\n{step_name} failed. Continue with next step? (y/n): ")
                    if continue_choice.lower() != 'y':
                        print("Pipeline stopped by user.")
                        break
        
        except KeyboardInterrupt:
            print(f"\n{step_name} interrupted by user.")
            results[step_name] = {'success': False, 'time': time.time() - start_time}
            break
        except Exception as e:
            print(f"\n✗ {step_name} error: {str(e)}")
            results[step_name] = {'success': False, 'time': time.time() - start_time, 'error': str(e)}
            
            # Ask if we should continue
            if step_name != "Web Scraping":
                continue_choice = input(f"\n{step_name} error. Continue with next step? (y/n): ")
                if continue_choice.lower() != 'y':
                    print("Pipeline stopped due to error.")
                    break
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    # Summary
    successful_steps = [name for name, result in results.items() if result.get('success', False)]
    failed_steps = [name for name, result in results.items() if not result.get('success', True)]
    
    print(f"\nSuccessful steps: {len(successful_steps)}/{len(steps)}")
    if successful_steps:
        print(f"  - {', '.join(successful_steps)}")
    
    if failed_steps:
        print(f"\nFailed steps: {len(failed_steps)}/{len(steps)}")
        print(f"  - {', '.join(failed_steps)}")
    
    total_time = sum(result.get('time', 0) for result in results.values())
    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"End time: {datetime.now().isoformat()}")
    
    return len(failed_steps) == 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Weather Anomaly Detection Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --setup                    Setup the project
  %(prog)s --pipeline                 Run the full data pipeline
  %(prog)s --dashboard                Start the dashboard
  %(prog)s --scrape                   Run web scraping only
  %(prog)s --process                  Run data preprocessing only
  %(prog)s --anomaly                  Run anomaly detection only
  %(prog)s --forecast                 Run forecasting only
  %(prog)s --schedule                 Schedule hourly scraping
        """
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Setup the project structure and check dependencies'
    )
    
    parser.add_argument(
        '--pipeline',
        action='store_true',
        help='Run the complete data pipeline (scrape → process → anomaly → forecast)'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start the Streamlit dashboard'
    )
    
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Run web scraping only'
    )
    
    parser.add_argument(
        '--process',
        action='store_true',
        help='Run data preprocessing only'
    )
    
    parser.add_argument(
        '--anomaly',
        action='store_true',
        help='Run anomaly detection only'
    )
    
    parser.add_argument(
        '--forecast',
        action='store_true',
        help='Run forecasting only'
    )
    
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Schedule hourly scraping (runs in background)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Setup, run pipeline, and start dashboard'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Run setup if requested
        if args.setup or args.all:
            if not setup_project():
                print("Setup failed. Please check the errors above.")
                return
        
        # Run complete pipeline
        if args.pipeline or args.all:
            success = run_full_pipeline()
            if not success:
                print("\nPipeline had some failures. Check logs for details.")
        
        # Run individual components
        if args.scrape:
            run_scraping()
        
        if args.process:
            run_preprocessing()
        
        if args.anomaly:
            run_anomaly()
        
        if args.forecast:
            run_forecast()
        
        # Schedule scraping
        if args.schedule:
            print("\nStarting hourly scraping scheduler...")
            print("Press Ctrl+C to stop")
            schedule_scraping()
        
        # Start dashboard
        if args.dashboard or args.all:
            print("\n" + "=" * 60)
            print("Starting Dashboard")
            print("=" * 60)
            print(f"Dashboard will be available at: http://localhost:{Config.DASHBOARD_PORT}")
            print("Press Ctrl+C to stop the dashboard")
            print("=" * 60)
            
            # Import and run streamlit
            import subprocess
            import signal
            
            try:
                # Run streamlit in a subprocess
                process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run",
                    "src/dashboard/app.py",
                    "--server.port", str(Config.DASHBOARD_PORT),
                    "--server.address", Config.DASHBOARD_HOST,
                    "--theme.base", "light"
                ])
                
                # Wait for process
                process.wait()
                
            except KeyboardInterrupt:
                print("\nDashboard stopped by user")
                if 'process' in locals():
                    process.terminate()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
