"""
Configuration settings for Weather Anomaly Detection Dashboard
"""

class Config:
    # Data paths
    RAW_DATA_PATH = "data/raw/weather_alerts_raw.csv"
    PROCESSED_DATA_PATH = "data/processed/weather_alerts_processed.csv"
    AGGREGATED_DATA_PATH = "data/processed/weather_alerts_aggregated.csv"
    ANOMALY_OUTPUT_PATH = "data/output/anomaly_results.csv"
    FORECAST_OUTPUT_PATH = "data/output/forecast_results.csv"
    
    # Model paths
    ANOMALY_MODEL_PATH = "models/isolation_forest.pkl"
    FORECAST_MODEL_PATH = "models/xgboost_forecast.pkl"
    
    # Scraping settings
    BASE_URL = "https://www.weather.gov"
    ALERTS_URL = "https://www.weather.gov/alerts"
    FORECAST_URL = "https://www.weather.gov/wrh/TextProduct"
    USER_AGENT = "WeatherAnomalyDetection/1.0 (Research Project)"
    SCRAPING_INTERVAL = 3600  # 1 hour in seconds
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Processing settings
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 10000
    STOPWORDS_LANGUAGE = "english"
    
    # ML settings
    ANOMALY_CONTAMINATION = 0.05
    FORECAST_HORIZON = 7
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Dashboard settings
    DASHBOARD_PORT = 8501
    DASHBOARD_HOST = "0.0.0.0"
    CACHE_TTL = 300  # 5 minutes
    MAX_DISPLAY_ROWS = 1000
    
    # Alert classification
    ALERT_TYPES = {
        'flood': ['flood', 'flash flood', 'flooding'],
        'storm': ['thunderstorm', 'storm', 'severe storm', 'tornado'],
        'winter': ['winter', 'snow', 'ice', 'blizzard', 'freezing'],
        'fire': ['fire', 'wildfire', 'red flag'],
        'wind': ['wind', 'high wind', 'wind advisory'],
        'heat': ['heat', 'excessive heat', 'heat advisory'],
        'cold': ['cold', 'freeze', 'frost', 'wind chill'],
        'coastal': ['coastal', 'surf', 'tsunami'],
        'air': ['air quality', 'smoke', 'dust'],
        'marine': ['marine', 'small craft', 'gale']
    }
    
    REGIONS = [
        'northeast', 'southeast', 'midwest', 'south', 'west',
        'northwest', 'southwest', 'central', 'eastern', 'western',
        'northern', 'southern'
    ]
    
    SEVERITY_KEYWORDS = {
        'warning': ['warning', 'emergency', 'dangerous', 'severe'],
        'watch': ['watch', 'possible', 'potential'],
        'advisory': ['advisory', 'caution', 'alert']
    }
