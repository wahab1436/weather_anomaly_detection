import pandas as pd
import numpy as np
import re
import string
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, List
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class WeatherDataPreprocessor:
    """Preprocesses weather alert data for ML and dashboard."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.weather_stop_words = {'weather', 'alert', 'warning', 'watch', 'advisory',
                                  'national', 'service', 'issued', 'effective', 'expires'}
        self.all_stop_words = self.stop_words.union(self.weather_stop_words)
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load and clean raw alert data."""
        try:
            df = pd.read_csv(filepath, parse_dates=['scraped_at', 'effective', 'expires', 'issued'])
            
            # Ensure required columns
            required_cols = ['title', 'description', 'region', 'alert_type', 'scraped_at']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in raw data: {missing_cols}")
                # Create missing columns with default values
                for col in missing_cols:
                    if col == 'title':
                        df['title'] = df.get('alert_type', 'Unknown Alert')
                    elif col == 'description':
                        df['description'] = ''
                    elif col == 'region':
                        df['region'] = 'Unknown'
                    elif col == 'alert_type':
                        df['alert_type'] = 'Other'
                    elif col == 'scraped_at':
                        df['scraped_at'] = datetime.now()
            
            # Clean data
            df = df.dropna(subset=['title', 'scraped_at'])
            df = df.fillna({
                'description': '',
                'region': 'Unknown',
                'alert_type': 'Other',
                'severity': 'Unknown'
            })
            
            # Convert text columns to string
            text_cols = ['title', 'description', 'region', 'alert_type', 'severity']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            logger.info(f"Loaded {len(df)} alerts from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and numbers (keep spaces and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.all_stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from alert data."""
        if df.empty:
            return pd.DataFrame()
        
        # Create features DataFrame
        features_df = pd.DataFrame()
        
        # Date features
        df['date'] = pd.to_datetime(df['scraped_at']).dt.date
        df['hour'] = pd.to_datetime(df['scraped_at']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['scraped_at']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Text features
        df['cleaned_title'] = df['title'].apply(self.preprocess_text)
        df['cleaned_description'] = df['description'].apply(self.preprocess_text)
        
        # Length features
        df['title_length'] = df['title'].str.len()
        df['desc_length'] = df['description'].str.len()
        df['has_description'] = (df['desc_length'] > 10).astype(int)
        
        # Keyword features
        keywords = {
            'flood': ['flood', 'flooding', 'flash flood'],
            'storm': ['storm', 'thunderstorm', 'lightning'],
            'wind': ['wind', 'gust', 'breez'],
            'rain': ['rain', 'precipitation', 'shower'],
            'snow': ['snow', 'blizzard', 'sleet'],
            'heat': ['heat', 'hot', 'temperature'],
            'cold': ['cold', 'freeze', 'frost'],
            'fire': ['fire', 'wildfire', 'burn'],
            'tornado': ['tornado', 'funnel', 'twister']
        }
        
        for keyword, patterns in keywords.items():
            pattern = '|'.join(patterns)
            df[f'has_{keyword}'] = df['cleaned_title'].str.contains(pattern).astype(int)
        
        # Aggregate by date and alert type
        daily_aggregated = self._aggregate_daily_counts(df)
        
        return daily_aggregated
    
    def _aggregate_daily_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate alerts by date and type."""
        if df.empty:
            return pd.DataFrame()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['scraped_at']).dt.date
        
        # Create pivot table
        pivot = pd.pivot_table(
            df,
            values='title',
            index='date',
            columns='alert_type',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        # Add total alerts
        pivot['total_alerts'] = pivot.select_dtypes(include=[np.number]).sum(axis=1)
        
        # Add region counts
        region_pivot = pd.pivot_table(
            df,
            values='title',
            index='date',
            columns='region',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        # Merge region data
        result = pd.merge(pivot, region_pivot, on='date', how='left', suffixes=('', '_region'))
        
        # Add date features
        result['date'] = pd.to_datetime(result['date'])
        result['day_of_week'] = result['date'].dt.dayofweek
        result['month'] = result['date'].dt.month
        result['year'] = result['date'].dt.year
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        
        # Fill NaN values
        result = result.fillna(0)
        
        return result
    
    def extract_tfidf_features(self, texts: List[str], max_features: int = 100) -> pd.DataFrame:
        """Extract TF-IDF features from text."""
        if not texts:
            return pd.DataFrame()
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(self.all_stop_words),
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Save vectorizer for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        
        return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    def prepare_forecast_data(self, aggregated_df: pd.DataFrame, target_col: str = 'total_alerts',
                            lookback: int = 7, forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare time series data for forecasting."""
        if aggregated_df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Ensure data is sorted by date
        df_sorted = aggregated_df.sort_values('date').reset_index(drop=True)
        
        features = []
        targets = []
        
        for i in range(lookback, len(df_sorted) - forecast_horizon):
            # Historical features
            hist_features = df_sorted.iloc[i-lookback:i].select_dtypes(include=[np.number]).values.flatten()
            
            # Date features for prediction day
            date_features = [
                df_sorted.iloc[i]['day_of_week'],
                df_sorted.iloc[i]['month'],
                df_sorted.iloc[i]['is_weekend']
            ]
            
            # Combine features
            combined_features = np.concatenate([hist_features, date_features])
            
            # Target (future value)
            target = df_sorted.iloc[i + forecast_horizon][target_col]
            
            features.append(combined_features)
            targets.append(target)
        
        features_df = pd.DataFrame(features)
        targets_series = pd.Series(targets)
        
        return features_df, targets_series
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """Save processed data to CSV."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")

def main():
    """Main preprocessing function."""
    preprocessor = WeatherDataPreprocessor()
    
    # Load raw data
    raw_filepath = 'data/raw/weather_alerts_raw.csv'
    if not os.path.exists(raw_filepath):
        logger.error(f"Raw data file not found: {raw_filepath}")
        return
    
    raw_df = preprocessor.load_raw_data(raw_filepath)
    
    if raw_df.empty:
        logger.warning("No data to process")
        return
    
    # Extract features
    processed_df = preprocessor.extract_features(raw_df)
    
    # Save processed data
    processed_path = 'data/processed/weather_alerts_processed.csv'
    preprocessor.save_processed_data(processed_df, processed_path)
    
    # Also save aggregated data for dashboard
    aggregated_path = 'data/output/daily_alert_counts.csv'
    daily_counts = processed_df[['date', 'total_alerts'] + 
                              [col for col in processed_df.columns if 'alert_type' in str(col)]].copy()
    daily_counts.to_csv(aggregated_path, index=False)
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()
