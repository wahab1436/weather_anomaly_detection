#!/usr/bin/env python3
"""
Streamlit-ready entry point for Weather Anomaly Detection System.
Wraps existing main.py backend functionality for cloud deployment.
"""
import streamlit as st
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import your backend modules
try:
    from scraping.scrape_weather_alerts import main as scrape_main
    from preprocessing.preprocess_text import preprocess_pipeline
    from ml.anomaly_detection import run_anomaly_detection
    from ml.forecast_model import run_forecasting
    from utils.helpers import setup_logging, generate_plain_english_insights, save_to_json, cleanup_old_files
except ModuleNotFoundError as e:
    st.error(f"Module import error: {e}")
    raise

# Setup logging
setup_logging('logs/system.log')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Weather Anomaly Detection", layout="wide")

st.title("Weather Anomaly Detection System")

# Sidebar controls
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
            "data/raw/weather_alerts_raw.csv",
            "data/processed/weather_alerts_processed.csv"
        )
        st.success("Preprocessing completed.")

        # Step 3: Anomaly detection
        st.write("Running anomaly detection...")
        run_anomaly_detection(
            "data/processed/weather_alerts_daily.csv",
            "data/output/anomaly_results.csv",
            "models/isolation_forest.pkl"
        )
        st.success("Anomaly detection completed.")

        # Step 4: Forecasting
        st.write("Running forecasting...")
        run_forecasting(
            "data/processed/weather_alerts_daily.csv",
            "data/output/forecast_results.csv",
            "models/xgboost_forecast.pkl"
        )
        st.success("Forecasting completed.")

        # Step 5: Generate insights
        st.write("Generating insights...")
        daily_stats_path = "data/processed/weather_alerts_daily.csv"
        anomalies_path = "data/output/anomaly_results.csv"
        forecasts_path = "data/output/forecast_results.csv"

        import pandas as pd
        daily_stats = pd.read_csv(daily_stats_path) if os.path.exists(daily_stats_path) else pd.DataFrame()
        anomalies = pd.read_csv(anomalies_path) if os.path.exists(anomalies_path) else pd.DataFrame()
        forecasts = pd.read_csv(forecasts_path) if os.path.exists(forecasts_path) else pd.DataFrame()

        insights = generate_plain_english_insights(daily_stats, anomalies, forecasts)
        save_to_json({
            "generated_at": datetime.now().isoformat(),
            "insights": insights
        }, "data/output/insights.json")

        st.success("Insights generated!")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")

# Sidebar: Cleanup
if st.sidebar.button("Cleanup Old Data"):
    st.info("Cleaning up old files...")
    try:
        cleanup_old_files('data/raw', days_to_keep=90)
        cleanup_old_files('logs', days_to_keep=30, pattern='*.log')
        st.success("Cleanup completed.")
    except Exception as e:
        st.error(f"Cleanup failed: {e}")

st.markdown("---")
st.info("Use the left sidebar to run pipelines or cleanup operations.")
