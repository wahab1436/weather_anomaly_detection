"""
Text preprocessing module for weather alerts.
Cleans and structures unstructured alert text.
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from typing import List, Dict
import json
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from textblob import TextBlob

# -------------------- NLTK SAFE DOWNLOAD (ONLY punkt) --------------------
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
        self.stop_words.update([
            "weather", "national", "service", "alert",
            "warning", "advisory", "watch", "issued"
        ])

        self.tokenizer = PunktSentenceTokenizer()

    # -------------------- TEXT CLEANING --------------------
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------------------- KEYWORDS (NO word_tokenize) --------------------
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        sentences = self.tokenizer.tokenize(text)
        tokens = []

        for sent in sentences:
            tokens.extend(sent.split())

        tokens = [
            t for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]

        from collections import Counter
        freq = Counter(tokens)
        return [w for w, _ in freq.most_common(top_n)]

    # -------------------- SENTIMENT --------------------
    def extract_sentiment(self, text: str) -> Dict:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.3:
            label = "urgent_negative"
        elif polarity > 0.1:
            label = "cautionary"
        elif polarity > -0.1:
            label = "neutral"
        elif polarity > -0.3:
            label = "concern"
        else:
            label = "severe"

        return {
            "sentiment_score": polarity,
            "sentiment_label": label,
            "subjectivity": blob.sentiment.subjectivity
        }

    # -------------------- ENTITIES --------------------
    def extract_entities(self, text: str) -> Dict:
        entities = {
            "locations": [],
            "measurements": [],
            "time_references": []
        }

        location_patterns = [
            r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(County|Parish|Borough))"
        ]

        for pat in location_patterns:
            entities["locations"].extend(re.findall(pat, text, re.IGNORECASE))

        entities["measurements"] = re.findall(
            r"(\d+(?:\.\d+)?)\s*(mph|inches|in|feet|ft|°f|°c|%)",
            text,
            re.IGNORECASE
        )

        entities["time_references"] = re.findall(
            r"(\d{1,2}:\d{2}\s*(AM|PM))",
            text,
            re.IGNORECASE
        )

        for k in entities:
            entities[k] = list(set(entities[k]))

        return entities

    # -------------------- METRICS --------------------
    def calculate_alert_metrics(self, text: str) -> Dict:
        words = text.split()

        return {
            "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "exclamation_count": text.count("!"),
            "numeric_count": len(re.findall(r"\d+", text))
        }

    # -------------------- MAIN DATAFRAME PROCESS --------------------
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["keywords"] = df["cleaned_text"].apply(self.extract_keywords)

        sentiment = df["cleaned_text"].apply(self.extract_sentiment)
        df = pd.concat([df, sentiment.apply(pd.Series)], axis=1)

        df["entities"] = df["text"].apply(self.extract_entities)

        metrics = df["text"].apply(self.calculate_alert_metrics)
        df = pd.concat([df, metrics.apply(pd.Series)], axis=1)

        for col in ["issued_date", "scraped_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "issued_date" in df.columns:
            df["hour"] = df["issued_date"].dt.hour
            df["day"] = df["issued_date"].dt.day
            df["month"] = df["issued_date"].dt.month
            df["year"] = df["issued_date"].dt.year

        logger.info(f"Preprocessed {len(df)} alerts")
        return df

    # -------------------- DAILY AGGREGATION --------------------
    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "issued_date" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df.set_index("issued_date", inplace=True)

        daily = df.resample("D").agg({
            "alert_id": "count",
            "sentiment_score": "mean",
            "word_count": "mean"
        }).rename(columns={"alert_id": "total_alerts"})

        daily.reset_index(inplace=True)
        return daily


# -------------------- PIPELINE --------------------
def preprocess_pipeline(input_path: str, output_path: str):
    try:
        if not os.path.exists(input_path):
            dates = pd.date_range(end=datetime.now(), periods=50, freq="H")
            df = pd.DataFrame({
                "alert_id": [f"sample_{i}" for i in range(50)],
                "title": ["Sample Alert"] * 50,
                "text": ["Sample weather alert text"] * 50,
                "type": ["storm"] * 50,
                "issued_date": dates,
                "scraped_at": datetime.now()
            })
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            df.to_csv(input_path, index=False)
        else:
            df = pd.read_csv(input_path)

        pre = WeatherAlertPreprocessor()
        processed = pre.preprocess_dataframe(df)
        daily = pre.create_daily_aggregates(processed)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("data/output", exist_ok=True)

        processed.to_csv(output_path, index=False)
        daily.to_csv("data/processed/weather_alerts_daily.csv", index=False)
        daily.to_csv("data/output/dashboard_data.csv", index=False)

        with open("data/output/insights.json", "w") as f:
            json.dump({
                "status": "success",
                "processed_records": len(processed),
                "daily_records": len(daily)
            }, f, indent=2)

        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    preprocess_pipeline(
        "data/raw/weather_alerts_raw.csv",
        "data/processed/weather_alerts_processed.csv"
    )
