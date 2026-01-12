"""
Script to collect initial historical data for the system.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scraping.scrape_weather_alerts import WeatherAlertScraper
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_historical_data(days_back: int = 30):
    """
    Collect historical data by simulating past dates.
    Note: This is a simulation. Real historical data would require
    accessing weather.gov's historical archives.
    """
    scraper = WeatherAlertScraper()
    
    all_alerts = []
    
    # For demonstration, we'll collect current data multiple times
    # In production, you would need to access historical archives
    logger.info(f"Collecting initial data (simulating {days_back} days)")
    
    for day in range(days_back):
        try:
            logger.info(f"Collecting data for day {day + 1}/{days_back}")
            
            # Scrape current alerts
            alerts = scraper.scrape_all_alerts()
            
            if alerts:
                # Adjust dates to simulate historical data
                for alert in alerts:
                    alert_date = alert['issued_date']
                    if isinstance(alert_date, datetime):
                        alert['issued_date'] = alert_date - timedelta(days=day)
                    else:
                        alert['issued_date'] = datetime.now() - timedelta(days=day)
                
                all_alerts.extend(alerts)
                logger.info(f"Collected {len(alerts)} alerts")
            
            # Be respectful with requests
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error collecting data for day {day}: {str(e)}")
    
    # Save collected data
    if all_alerts:
        df = pd.DataFrame(all_alerts)
        
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/output', exist_ok=True)
        
        # Save to raw data
        raw_path = 'data/raw/weather_alerts_raw.csv'
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved {len(df)} alerts to {raw_path}")
        
        # Also create a sample processed file
        sample_data = {
            'issued_date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
            'total_alerts': [10, 15, 8, 20, 12, 25, 18, 22, 14, 30, 
                            16, 24, 19, 28, 21, 17, 26, 23, 20, 15,
                            18, 22, 25, 19, 16, 21, 24, 27, 20, 18],
            'flood': [2, 3, 1, 4, 2, 5, 3, 4, 2, 6, 3, 5, 4, 6, 5, 3, 6, 5, 4, 3, 4, 5, 6, 4, 3, 5, 6, 7, 5, 4],
            'storm': [3, 4, 2, 5, 3, 7, 5, 6, 4, 8, 5, 7, 6, 8, 7, 5, 8, 7, 6, 5, 6, 7, 8, 6, 5, 7, 8, 9, 7, 6],
            'wind': [1, 2, 1, 3, 2, 4, 3, 4, 2, 5, 3, 4, 3, 5, 4, 3, 5, 4, 3, 2, 3, 4, 5, 3, 2, 4, 5, 6, 4, 3]
        }
        
        sample_df = pd.DataFrame(sample_data)
        processed_path = 'data/processed/weather_alerts_daily.csv'
        sample_df.to_csv(processed_path, index=False)
        logger.info(f"Created sample processed data at {processed_path}")
        
        return df
    
    return pd.DataFrame()

if __name__ == "__main__":
    # Collect initial data
    data = collect_historical_data(days_back=7)
    
    if not data.empty:
        print(f"\nInitial data collection complete!")
        print(f"Total alerts collected: {len(data)}")
        print(f"Date range: {data['issued_date'].min()} to {data['issued_date'].max()}")
        print(f"\nNext steps:")
        print("1. Run preprocessing: python main.py preprocess")
        print("2. Train models: python main.py detect-anomalies")
        print("3. Start dashboard: streamlit run src/dashboard/app.py")
    else:
        print("No data was collected. Check your connection and try again.")
