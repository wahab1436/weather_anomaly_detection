from src.scraping.scrape_weather_alerts import main as scrape
from src.preprocessing.preprocess_text import preprocess_pipeline
from src.ml.anomaly_detection import run_anomaly_detection
from src.ml.forecast_model import run_forecast

def main():
    scrape()  # live data only

    preprocess_pipeline(
        "data/raw/weather_alerts_raw.csv",
        "data/processed/weather_alerts_processed.csv"
    )

    run_anomaly_detection()
    run_forecast()

if __name__ == "__main__":
    main()
