"""
Text preprocessing module for weather alerts.
Cleans and structures unstructured alert text.
"""
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import json
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class WeatherAlertPreprocessor:
    """Preprocess weather alert text for analysis."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add weather-specific stop words
        self.stop_words.update(['weather', 'national', 'service', 'alert', 
                               'warning', 'advisory', 'watch', 'issued'])
        
        # Weather severity keywords
        self.severity_keywords = {
            'severe': ['severe', 'extreme', 'dangerous', 'emergency', 'catastrophic'],
            'moderate': ['moderate', 'significant', 'considerable'],
            'minor': ['minor', 'light', 'scattered', 'isolated']
        }
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and numbers (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Get frequency
        from collections import Counter
        word_freq = Counter(tokens)
        
        # Return top N keywords
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def extract_sentiment(self, text: str) -> Dict:
        """Extract sentiment from alert text."""
        if not text:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'subjectivity': 0.0
            }
        
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.3:
            sentiment = "urgent_negative"
        elif sentiment_score > 0.1:
            sentiment = "cautionary"
        elif sentiment_score > -0.1:
            sentiment = "neutral"
        elif sentiment_score > -0.3:
            sentiment = "concern"
        else:
            sentiment = "severe"
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text (locations, measurements)."""
        entities = {
            'locations': [],
            'measurements': [],
            'time_references': []
        }
        
        if not text:
            return entities
        
        # Extract locations (simple pattern matching)
        location_patterns = [
            r'in\s+([A-Z][a-z]+\s*(?:County|Parish|Borough))',
            r'for\s+([A-Z][a-z]+\s*(?:County|Parish|Borough))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:County|Parish|Borough))'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend(matches)
        
        # Extract measurements (numbers with units)
        measurement_pattern = r'(\d+(?:\.\d+)?)\s*(mph|inches|in|feet|ft|°F|°C|degrees|percent|%)'
        entities['measurements'] = re.findall(measurement_pattern, text, re.IGNORECASE)
        
        # Extract time references
        time_patterns = [
            r'until\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
            r'from\s+(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}\s*UTC)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities['time_references'].extend(matches)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def calculate_alert_metrics(self, text: str) -> Dict:
        """Calculate various metrics for an alert."""
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'all_caps_count': 0,
                'urgency_keywords': 0,
                'numeric_count': 0
            }
        
        # Text length metrics
        words = text.split()
        sentences = text.split('.')
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if len(s.strip()) > 0]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'exclamation_count': text.count('!'),
            'all_caps_count': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'urgency_keywords': sum(1 for word in ['immediate', 'urgent', 'emergency', 'warning'] 
                                  if word in text.lower()),
            'numeric_count': len(re.findall(r'\b\d+\b', text))
        }
        
        return metrics
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe of alerts."""
        if df.empty:
            logger.warning("Empty dataframe received for preprocessing")
            return df
        
        # Make a copy
        processed_df = df.copy()
        
        # Check if 'text' column exists, if not try to create it from other columns
        if 'text' not in processed_df.columns:
            logger.warning("'text' column not found. Attempting to create from available columns...")
            
            # Try to create text from title and description
            text_parts = []
            if 'title' in processed_df.columns:
                text_parts.append(processed_df['title'].fillna(''))
            if 'description' in processed_df.columns:
                text_parts.append(processed_df['description'].fillna(''))
            if 'headline' in processed_df.columns:
                text_parts.append(processed_df['headline'].fillna(''))
            
            if text_parts:
                processed_df['text'] = text_parts[0]
                for part in text_parts[1:]:
                    processed_df['text'] = processed_df['text'] + ' ' + part
                logger.info("Created 'text' column from available fields")
            else:
                # If no text-like columns exist, create empty text column
                processed_df['text'] = ''
                logger.warning("No text-like columns found. Using empty text for processing.")
        
        # Ensure text column is string type
        processed_df['text'] = processed_df['text'].fillna('').astype(str)
        
        # Clean text
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_text)
        
        # Extract keywords
        processed_df['keywords'] = processed_df['cleaned_text'].apply(
            lambda x: self.extract_keywords(x, top_n=5)
        )
        
        # Extract sentiment
        sentiment_data = processed_df['cleaned_text'].apply(self.extract_sentiment)
        processed_df = pd.concat([
            processed_df,
            sentiment_data.apply(pd.Series)
        ], axis=1)
        
        # Extract entities
        processed_df['entities'] = processed_df['text'].apply(self.extract_entities)
        
        # Calculate metrics
        metrics_data = processed_df['text'].apply(self.calculate_alert_metrics)
        processed_df = pd.concat([
            processed_df,
            metrics_data.apply(pd.Series)
        ], axis=1)
        
        # Parse dates
        date_columns = ['issued_date', 'scraped_at', 'effective', 'expires']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        
        # Use issued_date as primary date, fallback to scraped_at or other date columns
        if 'issued_date' not in processed_df.columns or processed_df['issued_date'].isna().all():
            if 'scraped_at' in processed_df.columns:
                processed_df['issued_date'] = processed_df['scraped_at']
            elif 'effective' in processed_df.columns:
                processed_df['issued_date'] = processed_df['effective']
            else:
                # Create a default date column if none exists
                processed_df['issued_date'] = pd.Timestamp.now()
                logger.warning("No valid date columns found. Using current timestamp.")
        
        # Extract time features
        if 'issued_date' in processed_df.columns and not processed_df['issued_date'].isna().all():
            processed_df['hour'] = processed_df['issued_date'].dt.hour
            processed_df['day_of_week'] = processed_df['issued_date'].dt.dayofweek
            processed_df['day_of_year'] = processed_df['issued_date'].dt.dayofyear
            processed_df['week_of_year'] = processed_df['issued_date'].dt.isocalendar().week
            processed_df['month'] = processed_df['issued_date'].dt.month
            processed_df['year'] = processed_df['issued_date'].dt.year
        
        # Create alert type groupings
        alert_type_mapping = {
            'flood': 'hydrological',
            'storm': 'meteorological',
            'wind': 'meteorological',
            'winter': 'meteorological',
            'fire': 'environmental',
            'heat': 'environmental',
            'cold': 'environmental',
            'coastal': 'oceanic',
            'air': 'environmental',
            'other': 'other'
        }
        
        # Ensure 'type' column exists
        if 'type' not in processed_df.columns:
            processed_df['type'] = 'other'
            logger.warning("'type' column not found. Setting all alerts to 'other'.")
        
        processed_df['alert_category'] = processed_df['type'].map(
            lambda x: alert_type_mapping.get(str(x).lower(), 'other')
        )
        
        # Create severity score
        processed_df['severity_score'] = processed_df.apply(
            lambda row: self._calculate_severity_score(row), axis=1
        )
        
        logger.info(f"Preprocessed {len(processed_df)} alerts")
        return processed_df
    
    def _calculate_severity_score(self, row) -> float:
        """Calculate a composite severity score."""
        score = 0.0
        
        # Base score from severity field
        severity_map = {
            'extreme': 1.0,
            'severe': 0.8,
            'moderate': 0.5,
            'minor': 0.2,
            'unknown': 0.1
        }
        
        if 'severity' in row:
            severity_str = str(row['severity']).lower()
            score += severity_map.get(severity_str, 0.1)
        else:
            score += 0.3  # Default score if no severity field
        
        # Add score from sentiment
        if 'sentiment_score' in row and not pd.isna(row['sentiment_score']):
            sentiment_score = row['sentiment_score']
            # Negative sentiment indicates higher severity
            score += abs(min(sentiment_score, 0)) * 0.5
        
        # Add score from urgency keywords
        if 'urgency_keywords' in row and not pd.isna(row['urgency_keywords']):
            score += min(row['urgency_keywords'] * 0.1, 0.3)
        
        # Add score from exclamation marks
        if 'exclamation_count' in row and not pd.isna(row['exclamation_count']):
            score += min(row['exclamation_count'] * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated statistics."""
        if df.empty:
            logger.warning("Empty dataframe received for daily aggregation")
            return pd.DataFrame()
        
        # Ensure date column exists
        if 'issued_date' not in df.columns:
            logger.error("No issued_date column found for aggregation")
            # Create a minimal daily stats DataFrame
            return self._create_minimal_daily_stats()
        
        # Remove rows with invalid dates
        df_valid = df.dropna(subset=['issued_date']).copy()
        
        if df_valid.empty:
            logger.warning("No valid dates found for aggregation")
            return self._create_minimal_daily_stats()
        
        # Set date as index for resampling
        df_date = df_valid.copy()
        df_date.set_index('issued_date', inplace=True)
        
        # Create aggregation dictionary
        agg_dict = {}
        
        # Add alert_id or create a count column
        if 'alert_id' in df_date.columns:
            agg_dict['alert_id'] = 'count'
        else:
            df_date['count'] = 1
            agg_dict['count'] = 'sum'
        
        # Add other numeric columns if they exist
        if 'severity_score' in df_date.columns:
            agg_dict['severity_score'] = 'mean'
        if 'sentiment_score' in df_date.columns:
            agg_dict['sentiment_score'] = 'mean'
        if 'word_count' in df_date.columns:
            agg_dict['word_count'] = 'mean'
        
        # Daily counts
        daily_counts = df_date.resample('D').agg(agg_dict)
        
        # Rename count column to total_alerts
        if 'alert_id' in daily_counts.columns:
            daily_counts.rename(columns={'alert_id': 'total_alerts'}, inplace=True)
        elif 'count' in daily_counts.columns:
            daily_counts.rename(columns={'count': 'total_alerts'}, inplace=True)
        else:
            daily_counts['total_alerts'] = 0
        
        # Count by alert type
        if 'type' in df_date.columns:
            alert_type_dummies = pd.get_dummies(df_date['type'])
            daily_type_counts = alert_type_dummies.resample('D').sum()
            
            # Combine all daily stats
            daily_stats = pd.concat([daily_counts, daily_type_counts], axis=1)
        else:
            daily_stats = daily_counts
        
        # Fill NaN values
        daily_stats = daily_stats.fillna(0)
        
        # Add derived metrics
        if 'severity_score' in daily_stats.columns and 'total_alerts' in daily_stats.columns:
            daily_stats['alert_intensity'] = daily_stats['total_alerts'] * daily_stats['severity_score']
        
        # Add rolling statistics
        if 'total_alerts' in daily_stats.columns:
            daily_stats['7_day_avg'] = daily_stats['total_alerts'].rolling(window=7, min_periods=1).mean()
            daily_stats['30_day_avg'] = daily_stats['total_alerts'].rolling(window=30, min_periods=1).mean()
            
            # Calculate day-over-day change
            daily_stats['day_over_day_change'] = daily_stats['total_alerts'].pct_change() * 100
        
        # Reset index for easier handling
        daily_stats.reset_index(inplace=True)
        
        logger.info(f"Created daily aggregates for {len(daily_stats)} days")
        return daily_stats
    
    def _create_minimal_daily_stats(self) -> pd.DataFrame:
        """Create a minimal daily stats DataFrame with sample data."""
        logger.info("Creating minimal daily stats with sample data")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        return pd.DataFrame({
            'issued_date': dates,
            'total_alerts': np.random.randint(5, 30, 30),
            'severity_score': np.random.uniform(0.2, 0.8, 30),
            'flood': np.random.randint(0, 5, 30),
            'storm': np.random.randint(0, 10, 30),
            'wind': np.random.randint(0, 8, 30),
            'winter': np.random.randint(0, 5, 30),
            'other': np.random.randint(0, 5, 30)
        })

def preprocess_pipeline(input_path: str, output_path: str):
    """Complete preprocessing pipeline."""
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found: {input_path}. Creating sample data...")
            
            # Create sample data
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            sample_df = pd.DataFrame({
                'alert_id': range(1, 51),
                'title': [f'Weather Alert {i}' for i in range(1, 51)],
                'description': [f'Sample weather alert description {i}' for i in range(1, 51)],
                'type': np.random.choice(['flood', 'storm', 'wind', 'winter', 'other'], 50),
                'severity': np.random.choice(['minor', 'moderate', 'severe'], 50),
                'issued_date': pd.date_range(end=datetime.now(), periods=50, freq='H')
            })
            sample_df.to_csv(input_path, index=False)
            logger.info(f"Created sample data at {input_path}")
        
        # Load raw data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        
        # Initialize preprocessor
        preprocessor = WeatherAlertPreprocessor()
        
        # Preprocess data
        processed_df = preprocessor.preprocess_dataframe(df)
        
        # Create daily aggregates
        daily_stats = preprocessor.create_daily_aggregates(processed_df)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save individual alerts
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed alerts to {output_path}")
        
        # Save daily aggregates
        daily_output_path = output_path.replace('_processed.csv', '_daily.csv')
        if not daily_output_path.endswith('_daily.csv'):
            daily_output_path = output_path.replace('.csv', '_daily.csv')
        
        daily_stats.to_csv(daily_output_path, index=False)
        logger.info(f"Saved daily aggregates to {daily_output_path}")
        
        # Save to output folder for dashboard
        os.makedirs("data/output", exist_ok=True)
        dashboard_output = "data/output/dashboard_data.csv"
        daily_stats.to_csv(dashboard_output, index=False)
        logger.info(f"Saved dashboard data to {dashboard_output}")
        
        logger.info("Preprocessing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/weather_alerts_raw.csv"
    output_file = "data/processed/weather_alerts_processed.csv"
    
    preprocess_pipeline(input_file, output_file)
