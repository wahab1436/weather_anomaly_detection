"""
Weather Anomaly Detection Dashboard - Main Package
"""

__version__ = "1.0.0"
__author__ = "Weather Anomaly Detection Team"

from src.scraping.scrape_weather_alerts import WeatherAlertScraper
from src.preprocessing.preprocess_text import TextPreprocessor
from src.ml.anomaly_detection import AnomalyDetector
from src.ml.forecast_model import AlertForecaster
from src.dashboard.app import WeatherAnomalyDashboard

__all__ = [
    'WeatherAlertScraper',
    'TextPreprocessor', 
    'AnomalyDetector',
    'AlertForecaster',
    'WeatherAnomalyDashboard'
]
