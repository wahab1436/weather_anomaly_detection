#!/usr/bin/env python3
"""
Streamlit-ready entry point for the Weather Anomaly Detection System.
Runs the dashboard from src/dashboard/app.py and keeps backend modules connected.
"""
import streamlit as st
import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add src to Python path
SRC_PATH = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_PATH)

# Import the dashboard app
try:
    from dashboard import app as dashboard_app
except ModuleNotFoundError as e:
    st.error(f"Dashboard module import error: {e}")
    raise

# Import backend modules
try:
    from scraping.scrape_weather_alerts import main as scrape_main
    from preprocessing.preprocess_text import preprocess_pipeline
    from ml.anomaly_detection import run_anomaly_detection
    from ml.forecast_model import run_forecasting
    from utils.helpers import (
        setup_logging, 
        generate_plain_english_insights, 
        save_to_json, 
        cleanup_old_files
    )
except ModuleNotFoundError as e:
    st.error(f"Backend module import error: {e}")
    raise

# Setup logging
setup_logging(os.path.join(SRC_PATH, '../logs/system.log'))
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    layout="wide"
)

# Sidebar for pipeline controls
st.sidebar.header("Pipeline Controls")

if st.sidebar.button("Run Full Pipeline"):
    st.info("Running full pipeline...")
    try:
        # Step 1: Scrape
        st.write("Starting scraping...")
        scrape_main()
        st.success("Scraping completed.")

        # Step 2: Preprocess
        st.write("Preprocessing data...")
        preprocess_pipeline(
            os.path.join(SRC_PATH, '../data/raw/weather_alerts_raw.csv'),
            os.path.join(SRC_PATH, '../data/processed/weather_alerts_processed.csv')
        )
        st.success("Preprocessing completed.")

        # Step 3: Anomaly detection
        st.write("Running anomaly detection...")
        run_anomaly_detection(
            os.path.join(SRC_PATH, '../data/processed/weather_alerts_daily.csv'),
            os.path.join(SRC_PATH, '../data/output/anomaly_results.csv'),
            os.path.join(SRC_PATH, '../models/isolation_forest.pkl')
        )
        st.success("Anomaly detection completed.")

        # Step 4: Forecasting
        st.write("Running forecasting...")
        run_forecasting(
            os.path.join(SRC_PATH, '../data/processed/weather_alerts_daily.csv'),
            os.path.join(SRC_PATH, '../data/output/forecast_results.csv'),
            os.path.join(SRC_PATH, '../models/xgboost_forecast.pkl')
        )
        st.success("Forecasting completed.")

        # Step 5: Generate insights
        st.write("Generating insights...")
        
        daily_stats_path = os.path.join(SRC_PATH, '../data/processed/weather_alerts_daily.csv')
        anomalies_path = os.path.join(SRC_PATH, '../data/output/anomaly_results.csv')
        forecasts_path = os.path.join(SRC_PATH, '../data/output/forecast_results.csv')

        daily_stats = pd.read_csv(daily_stats_path) if os.path.exists(daily_stats_path) else pd.DataFrame()
        anomalies = pd.read_csv(anomalies_path) if os.path.exists(anomalies_path) else pd.DataFrame()
        forecasts = pd.read_csv(forecasts_path) if os.path.exists(forecasts_path) else pd.DataFrame()

        insights = generate_plain_english_insights(daily_stats, anomalies, forecasts)
        save_to_json({
            "generated_at": datetime.now().isoformat(),
            "insights": insights
        }, os.path.join(SRC_PATH, '../data/output/insights.json'))

        st.success("Insights generated!")
        st.success("Full pipeline completed successfully!")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

# Sidebar: Cleanup
if st.sidebar.button("Cleanup Old Data"):
    st.info("Cleaning up old files...")
    try:
        cleanup_old_files(os.path.join(SRC_PATH, '../data/raw'), days_to_keep=90)
        cleanup_old_files(os.path.join(SRC_PATH, '../logs'), days_to_keep=30, pattern='*.log')
        st.success("Cleanup completed.")
    except Exception as e:
        st.error(f"Cleanup failed: {e}")

# Run the dashboard app from src/dashboard/app.py
st.markdown("---")
st.info("Launching the Weather Anomaly Detection Dashboard below...")
dashboard_app.main()
