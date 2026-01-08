#!/usr/bin/env python3
"""
Simple runner for Weather Anomaly Detection Dashboard
Run this one file to set everything up
"""
import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("=" * 70)
    print("WEATHER ANOMALY DETECTION DASHBOARD")
    print("=" * 70)

def create_directories():
    """Create all required directories"""
    dirs = ['data/raw', 'data/processed', 'data/output', 'models', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Created all directories")

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'requests', 'beautifulsoup4', 'scikit-learn', 'xgboost']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"✗ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies installed")
        return True

def run_pipeline():
    """Run the complete data pipeline"""
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE DATA PIPELINE")
    print("=" * 70)
    
    # Import here to avoid issues
    try:
        # Scraping
        print("\n[1/4] Scraping weather alerts...")
        from src.scraping.scrape_weather_alerts import WeatherAlertScraper
        scraper = WeatherAlertScraper()
        count = scraper.run_scraping_job()
        print(f"   Scraped {count} alerts")
        
        # Preprocessing
        print("\n[2/4] Processing data...")
        from src.preprocessing.preprocess_text import run_preprocessing_pipeline
        df_proc, df_agg = run_preprocessing_pipeline()
        if df_proc is not None:
            print(f"   Processed {len(df_proc)} records")
        
        # Anomaly detection
        print("\n[3/4] Running anomaly detection...")
        from src.ml.anomaly_detection import run_anomaly_detection
        df_anomalies = run_anomaly_detection()
        if df_anomalies is not None:
            anomalies = df_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_anomalies.columns else 0
            print(f"   Detected {anomalies} anomalies")
        
        # Forecasting
        print("\n[4/4] Running forecasting...")
        from src.ml.forecast_model import run_forecast_pipeline
        forecasts, metrics = run_forecast_pipeline()
        if forecasts is not None:
            print(f"   Generated {len(forecasts)} days of forecasts")
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE! Now run: python run_me.py --dashboard")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def run_dashboard():
    """Start the Streamlit dashboard"""
    print("\n" + "=" * 70)
    print("STARTING DASHBOARD")
    print("=" * 70)
    print("The dashboard will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 70)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except Exception as e:
        print(f"Dashboard error: {e}")

def main():
    """Main function"""
    print_banner()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = input("\nChoose an option:\n1. Setup and run pipeline\n2. Just start dashboard\n3. Exit\n\nEnter choice (1-3): ")
        if command == "1":
            command = "--pipeline"
        elif command == "2":
            command = "--dashboard"
        else:
            return
    
    # Create directories first
    create_directories()
    
    if command in ["--setup", "--pipeline", "-p"]:
        if check_dependencies():
            run_pipeline()
            
            # Ask if user wants to start dashboard
            if input("\nStart dashboard now? (y/n): ").lower() == 'y':
                run_dashboard()
    
    elif command in ["--dashboard", "-d"]:
        run_dashboard()
    
    elif command in ["--scrape", "-s"]:
        from src.scraping.scrape_weather_alerts import WeatherAlertScraper
        scraper = WeatherAlertScraper()
        count = scraper.run_scraping_job()
        print(f"Scraped {count} alerts")
    
    elif command in ["--help", "-h"]:
        print("\nAvailable commands:")
        print("  --pipeline, -p    Run complete data pipeline (scrape + process + ML)")
        print("  --dashboard, -d   Start the dashboard")
        print("  --scrape, -s      Just scrape data")
        print("  --help, -h        Show this help")
    
    else:
        print(f"Unknown command: {command}")
        print("Use --help to see available commands")

if __name__ == "__main__":
    main()
