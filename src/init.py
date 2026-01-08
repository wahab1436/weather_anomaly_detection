"""
Weather Anomaly Detection Dashboard - Main Package
"""

__version__ = "1.0.0"
__author__ = "Weather Anomaly Team"

from src.scraping import WeatherAlertScraper
from src.preprocessing import TextPreprocessor
from src.ml import AnomalyDetector, AlertForecaster
from src.dashboard import WeatherAnomalyDashboard
