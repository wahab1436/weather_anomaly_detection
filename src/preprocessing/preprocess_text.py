"""
Text preprocessing module for weather alerts.
Cleans and structures unstructured alert text.
"""

import os
import json
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =========================
# NLTK SETUP (ONLY punkt)
# =========================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

logger = logging.getLogger(__name__)


class WeatherAlertPreprocessor:
    """Preprocess weather alert text for analysis."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(
            [
                "weather",
                "national",
                "service",
                "alert",
                "warning",
                "advisory",
                "watch",
                "issued",
            ]
        )

    # -------------------------
    # TEXT CLEANING
    # -------------------------
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------------------------
    # KEYWORDS
    # -------------------------
    def extract_keywords(self, text: str, top_n: int = 5):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        from collections import Counter

        return [w for w, _ in Counter(tokens).most_common(top_n)]

    # -------------------------
    # SENTIMENT (NO TextBlob)
    # -------------------------
    def extract_sentiment(self, text: str):
        negative = ["severe", "extreme", "danger", "emergency", "warning"]
        positive = ["minor", "light", "moderate"]

        score = 0.0
        for w in negative:
            if w in text:
                score -= 0.2
        for w in positive:
            if w in text:
                score += 0.1

        return {
            "sentiment_score": max(min(score, 1.0), -1.0),
            "subjectivity": 0.5,
        }

    # -------------------------
    # METRICS
    # -------------------------
    def calculate_alert_metrics(self, text: str):
        words = text.split()
        return {
            "word_count": len(words),
            "exclamation_count": text.count("!"),
            "numeric_count": len(re.findall(r"\d+", text)),
        }

    # -------------------------
    # MAIN DF PIPELINE
    # -------------------------
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["keywords"] = df["cleaned_text"].apply(self.extract_keywords)

        sentiment = df["cleaned_text"].apply(self.extract_sentiment)
        df = pd.concat([df, sentiment.apply(pd.Series)], axis=1)

        metrics = df["cleaned_text"].apply(self.calculate_alert_metrics)
        df = pd.concat([df, metrics.apply(pd.Series)], axis=1)

        if "issued_date" in df.columns:
            df["issued_date"] = pd.to_datetime(df["issued_date"], errors="coerce")

        logger.info(f"Preprocessed {len(df)} alerts")
        return df

    # -------------------------
    # DAILY AGGREGATES
    # -------------------------
    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "issued_date" not in df.columns:
            return pd.DataFrame()

        df = df.set_index("issued_date")
        daily = df.resample("D").agg(
            total_alerts=("alert_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_words=("word_count", "mean"),
        )
        daily.reset_index(inplace=True)
        return daily


# =========================
# PIPELINE FUNCTION
# =========================
def preprocess_pipeline(input_path: str, output_path: str):
    try:
        if not os.path.exists(input_path):
            os.makedirs(os.path.dirname(input_path), exist_ok=True)

            dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
            df = pd.DataFrame(
                {
                    "alert_id": [f"id_{i}" for i in range(100)],
                    "text": [f"Sample weather alert {i}" for i in range(100)],
                    "type": np.random.choice(["flood", "storm", "wind"], 100),
                    "issued_date": dates,
                }
            )
            df.to_csv(input_path, index=False)
        else:
            df = pd.read_csv(input_path)

        processor = WeatherAlertPreprocessor()
        processed = processor.preprocess_dataframe(df)
        daily = processor.create_daily_aggregates(processed)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("data/output", exist_ok=True)

        processed.to_csv(output_path, index=False)
        daily.to_csv("data/output/weather_alerts_daily.csv", index=False)

        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    preprocess_pipeline(
        "data/raw/weather_alerts_raw.csv",
        "data/processed/weather_alerts_processed.csv",
    )
