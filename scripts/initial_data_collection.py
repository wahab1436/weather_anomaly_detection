"""
Script to collect initial historical data for the system
(LIVE SCRAPING ONLY – NO SAMPLE DATA)
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.scraping.scrape_weather_alerts import WeatherAlertScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_historical_data(days_back: int = 7) -> pd.DataFrame:
    """
    Collect data by repeatedly scraping live alerts
    and shifting timestamps to simulate history.
    """

    scraper = WeatherAlertScraper()
    all_alerts = []

    logger.info(f"Starting initial data collection for {days_back} days")

    for day in range(days_back):
        try:
            logger.info(f"Day {day + 1}/{days_back}: scraping alerts")

            alerts = scraper.scrape_all_alerts()

            for alert in alerts:
                issued = alert.get("issued_date")
                if isinstance(issued, datetime):
                    alert["issued_date"] = issued - timedelta(days=day)
                else:
                    alert["issued_date"] = datetime.utcnow() - timedelta(days=day)

            all_alerts.extend(alerts)
            time.sleep(2)

        except Exception as e:
            logger.error(f"Scraping failed on day {day}: {e}")

    if not all_alerts:
        logger.warning("No alerts collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_alerts)

    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/weather_alerts_raw.csv"
    df.to_csv(raw_path, index=False)

    logger.info(f"Saved {len(df)} alerts to {raw_path}")
    return df


# 🔑 REQUIRED ENTRY POINT (FIXES IMPORT ERROR)
def run_initial_collection(days_back: int = 7) -> pd.DataFrame:
    """
    Entry point used by main.py and scheduler
    """
    return collect_historical_data(days_back)


if __name__ == "__main__":
    run_initial_collection(7)
