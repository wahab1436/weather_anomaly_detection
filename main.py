"""
Main entry point for Weather Alert Anomaly Detection System
Dashboard is the final entry point.
"""

import logging
import os

# -------------------- Scripts --------------------
from scripts.initial_data_collection import run_initial_collection
from scripts.run_scheduler import run_scheduler

# -------------------- Core Pipeline --------------------
from src.scraping.scrape_weather_alerts import main as scrape_weather_alerts
from src.preprocessing.preprocess_text import preprocess_pipeline
from src.ml.anomaly_detection import run_anomaly_detection
from src.ml.forecast_model import run_forecast

# -------------------- Dashboard --------------------
from src.dashboard.app import run_dashboard

# -------------------- Utils --------------------
from src.utils.helpers import setup_logging

# -------------------- Paths --------------------
RAW_DATA_PATH = "data/raw/weather_alerts_raw.csv"
PROCESSED_DATA_PATH = "data/processed/weather_alerts_processed.csv"


def main():
    """
    Full system runner:
    - Live scraping
    - Preprocessing
    - ML analysis
    - Dashboard launch
    """

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Weather Alert System")

    # -------------------- Step 1: Initial / Scheduled Scripts --------------------
    try:
        logger.info("Running initial data collection scripts...")
        run_initial_collection()
        run_scheduler()
    except Exception as e:
        logger.warning(f"Scripts skipped or failed: {e}")

    # -------------------- Step 2: Live Scraping --------------------
    logger.info("Starting live weather alert scraping...")
    scraped_count = scrape_weather_alerts()

    if scraped_count == 0:
        logger.error("No alerts scraped. Stopping pipeline.")
        return

    # -------------------- Step 3: Preprocessing --------------------
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Raw scraped data not found. Scraping failed.")

    logger.info("Starting preprocessing...")
    processed_df, daily_df = preprocess_pipeline(
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH
    )

    # -------------------- Step 4: ML Models --------------------
    logger.info("Running anomaly detection...")
    run_anomaly_detection(daily_df)

    logger.info("Running forecasting model...")
    run_forecast(daily_df)

    # -------------------- Step 5: Dashboard --------------------
    logger.info("Launching dashboard...")
    run_dashboard()


if __name__ == "__main__":
    main()
