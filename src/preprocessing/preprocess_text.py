"""
Text preprocessing module for weather alerts.
Cleans and structures unstructured alert text.
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob

# -------------------------------------------------------------------
# NLTK SAFE SETUP (ONLY punkt + stopwords, NO punkt_tab)
# -------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------


class WeatherAlertPreprocessor:
    """Preprocess weather alert text for analysis."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(
            [
                "weather",
                "service",
                "national",
                "alert",
                "warning",
                "advisory",
                "watch",
                "issued",
            ]
        )

    # ---------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------------------------------------------------------------

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        tokens = word_tokenize(text)
        tokens = [
            t for t in tokens if t not in self.stop_words and len(t) > 2
        ]

        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    # ---------------------------------------------------------------

    def extract_sentiment(self, text: str) -> Dict:
        blob = TextBlob(text)
        return {
            "sentiment_score": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    # ---------------------------------------------------------------

    def calculate_metrics(self, text: str) -> Dict:
        words = text.split()
        return {
            "word_count": len(words),
            "exclamation_count": text.count("!"),
            "numeric_count": len(re.findall(r"\d+", text)),
        }

    # ---------------------------------------------------------------

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["keywords"] = df["cleaned_text"].apply(self.extract_keywords)

        sentiment_df = df["cleaned_text"].apply(self.extract_sentiment).apply(
            pd.Series
        )
        df = pd.concat([df, sentiment_df], axis=1)

        metrics_df = df["cleaned_text"].apply(self.calculate_metrics).apply(
            pd.Series
        )
        df = pd.concat([df, metrics_df], axis=1)

        if "issued_date" in df.columns:
            df["issued_date"] = pd.to_datetime(df["issued_date"], errors="coerce")
            df["day"] = df["issued_date"].dt.date

        logger.info(f"Preprocessed {len(df)} alerts")
        return df

    # ---------------------------------------------------------------

    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "issued_date" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df.set_index("issued_date", inplace=True)

        daily = df.resample("D").agg(
            {
                "alert_id": "count",
                "sentiment_score": "mean",
                "word_count": "mean",
            }
        )

        daily.rename(columns={"alert_id": "total_alerts"}, inplace=True)
        daily.reset_index(inplace=True)

        return daily


# -------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------

def preprocess_pipeline(input_path: str, output_path: str):
    try:
        if not os.path.exists(input_path):
            logger.warning("Input file missing, creating sample data")

            dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
            df = pd.DataFrame(
                {
                    "alert_id": [f"a{i}" for i in range(100)],
                    "text": [
                        f"Sample weather alert message {i}" for i in range(100)
                    ],
                    "type": np.random.choice(
                        ["flood", "storm", "wind"], size=100
                    ),
                    "issued_date": dates,
                }
            )

            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            df.to_csv(input_path, index=False)
        else:
            df = pd.read_csv(input_path)

        preprocessor = WeatherAlertPreprocessor()

        processed_df = preprocessor.preprocess_dataframe(df)
        daily_df = preprocessor.create_daily_aggregates(processed_df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("data/output", exist_ok=True)

        processed_df.to_csv(output_path, index=False)
        daily_df.to_csv(
            "data/processed/weather_alerts_daily.csv", index=False
        )
        daily_df.to_csv("data/output/dashboard_data.csv", index=False)

        with open("data/output/insights.json", "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "alerts_processed": len(processed_df),
                },
                f,
                indent=2,
            )

        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Preprocessing pipeline failed: {e}")
        raise


# -------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_pipeline(
        "data/raw/weather_alerts_raw.csv",
        "data/processed/weather_alerts_processed.csv",
    )
