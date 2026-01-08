#!/usr/bin/env python3
"""
Command Line Interface for Weather Anomaly Detection Dashboard
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """CLI main function"""
    parser = argparse.ArgumentParser(
        description="Weather Anomaly Detection Dashboard - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('--setup', action='store_true', help='Setup project')
    parser.add_argument('--pipeline', action='store_true', help='Run full pipeline')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard')
    parser.add_argument('--scrape', action='store_true', help='Run scraping')
    parser.add_argument('--process', action='store_true', help='Run preprocessing')
    parser.add_argument('--anomaly', action='store_true', help='Run anomaly detection')
    parser.add_argument('--forecast', action='store_true', help='Run forecasting')
    parser.add_argument('--schedule', action='store_true', help='Schedule scraping')
    
    args = parser.parse_args()
    
    # Import here to avoid Streamlit issues
    from run_dashboard import (
        setup_project, run_full_pipeline, run_scraping,
        run_preprocessing, run_anomaly, run_forecast,
        schedule_scraping, start_dashboard
    )
    
    # Execute based on arguments
    if args.setup:
        setup_project()
    elif args.pipeline:
        run_full_pipeline()
    elif args.dashboard:
        start_dashboard()
    elif args.scrape:
        run_scraping()
    elif args.process:
        run_preprocessing()
    elif args.anomaly:
        run_anomaly()
    elif args.forecast:
        run_forecast()
    elif args.schedule:
        schedule_scraping()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
