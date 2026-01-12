"""
Main backend entry point for Weather Anomaly Detection project.
Handles scraping, preprocessing, anomaly detection, forecasts, and insights.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Directories ------------------
def setup_directories():
    """Ensure all necessary directories exist."""
    dirs = ['data/raw', 'data/processed', 'data/output', 'models', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
setup_directories()

# ------------------ Dummy Data Pipeline ------------------
def run_pipeline():
    """
    Full backend pipeline:
    1. Scrape data
    2. Preprocess
    3. Detect anomalies
    4. Forecast
    5. Generate insights
    """

    # --- Scrape (simulate) ---
    raw_file = 'data/raw/weather_alerts_raw.csv'
    if not os.path.exists(raw_file):
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        raw_data = pd.DataFrame({
            'issued_date': dates,
            'title': [f"Weather Alert {i}" for i in range(30)],
            'type': np.random.choice(['flood', 'storm', 'wind'], 30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30),
            'severity_score': np.random.uniform(0.1, 1.0, 30)
        })
        raw_data.to_csv(raw_file, index=False)
        logger.info(f"Raw data generated: {raw_file}")
    else:
        raw_data = pd.read_csv(raw_file)
        logger.info(f"Loaded raw data: {raw_file}")

    # --- Preprocess ---
    processed_file = 'data/processed/weather_alerts_daily.csv'
    if not os.path.exists(processed_file):
        daily_stats = raw_data.groupby('issued_date').agg(
            total_alerts=('title', 'count'),
            severity_score=('severity_score', 'mean'),
            flood=('type', lambda x: sum(x=='flood')),
            storm=('type', lambda x: sum(x=='storm')),
            wind=('type', lambda x: sum(x=='wind'))
        ).reset_index()
        daily_stats.to_csv(processed_file, index=False)
        logger.info(f"Processed daily stats saved: {processed_file}")
    else:
        daily_stats = pd.read_csv(processed_file)
        logger.info(f"Loaded processed daily stats: {processed_file}")

    # --- Anomaly Detection (simulate) ---
    anomaly_file = 'data/output/anomaly_results.csv'
    if not os.path.exists(anomaly_file):
        anomalies = daily_stats.copy()
        anomalies['is_anomaly'] = False
        anomaly_indices = np.random.choice(len(anomalies), 3, replace=False)
        anomalies.loc[anomaly_indices, 'is_anomaly'] = True
        anomalies.loc[anomaly_indices, 'anomaly_severity'] = np.random.choice(['low','medium','high'],3)
        anomalies.to_csv(anomaly_file, index=False)
        logger.info(f"Anomalies generated: {anomaly_file}")
    else:
        anomalies = pd.read_csv(anomaly_file)
        logger.info(f"Loaded anomalies: {anomaly_file}")

    # --- Forecasts (simulate) ---
    forecast_file = 'data/output/forecast_results.csv'
    if not os.path.exists(forecast_file):
        last_date = pd.to_datetime(daily_stats['issued_date']).max()
        forecast_dates = pd.date_range(start=last_date+timedelta(days=1), periods=7, freq='D')
        forecast = pd.DataFrame({
            'date': forecast_dates,
            'target': 'total_alerts',
            'forecast': np.random.randint(10, 40, 7),
            'lower_bound': np.random.randint(5, 35, 7),
            'upper_bound': np.random.randint(15, 45, 7)
        })
        forecast.to_csv(forecast_file, index=False)
        logger.info(f"Forecast generated: {forecast_file}")
    else:
        forecast = pd.read_csv(forecast_file)
        logger.info(f"Loaded forecast: {forecast_file}")

    # --- Insights (simulate) ---
    insights_file = 'data/output/insights.json'
    if not os.path.exists(insights_file):
        insights_data = {
            'generated_at': datetime.now().isoformat(),
            'insights': [
                "Backend pipeline completed successfully.",
                f"Processed {len(daily_stats)} days of alerts.",
                f"Detected {anomalies['is_anomaly'].sum()} anomalies.",
                "Forecast ready for next 7 days."
            ]
        }
        with open(insights_file, 'w') as f:
            json.dump(insights_data, f, indent=2)
        logger.info(f"Insights generated: {insights_file}")
    else:
        logger.info(f"Insights file exists: {insights_file}")

    logger.info("Backend pipeline finished successfully.")


# ------------------ Main ------------------
def main():
    logger.info("Running full backend pipeline...")
    run_pipeline()
    logger.info("Backend is ready. Use dashboard separately.")


if __name__ == "__main__":
    main()
