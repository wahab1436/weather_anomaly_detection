"""
Text preprocessing module for weather alerts
"""
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Preprocess and extract features from text data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words(Config.STOPWORDS_LANGUAGE))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Enhanced weather keywords
        self.weather_keywords = {
            'temperature': ['temp', 'temperature', '°f', '°c', 'degrees', 'warm', 'cold', 'hot'],
            'precipitation': ['rain', 'snow', 'precipitation', 'shower', 'drizzle', 'sleet'],
            'wind': ['wind', 'breeze', 'gust', 'mph', 'kph', 'knots'],
            'pressure': ['pressure', 'barometer', 'hg', 'mb', 'hpa'],
            'visibility': ['visibility', 'fog', 'mist', 'haze', 'clear'],
            'humidity': ['humidity', 'dewpoint', 'moisture'],
            'cloud': ['cloud', 'overcast', 'cloudy', 'sunny', 'clear'],
            'storm': ['thunderstorm', 'lightning', 'thunder', 'hail'],
            'flood': ['flood', 'flooding', 'inundation', 'water'],
            'fire': ['fire', 'wildfire', 'burn', 'smoke']
        }
        
        # Severity patterns
        self.severity_patterns = {
            'extreme': r'\b(extreme|catastrophic|devastating|life.?threatening)\b',
            'severe': r'\b(severe|dangerous|hazardous|emergency)\b',
            'moderate': r'\b(moderate|significant|considerable)\b',
            'minor': r'\b(minor|light|small|slight)\b'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep weather-related symbols
        text = re.sub(r'[^\w\s°\-%\"/\.:]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_numerical_features(self, text: str) -> Dict:
        """Extract numerical values from text"""
        features = {}
        
        # Temperature patterns
        temp_patterns = [
            r'(\d+)\s*°\s*f',  # Fahrenheit
            r'(\d+)\s*°\s*c',  # Celsius
            r'temp(?:erature)?\s*[:\.]?\s*(\d+)',
            r'(\d+)\s*degrees'
        ]
        
        for pattern in temp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                features['temperature'] = float(matches[0])
                break
        
        # Wind speed patterns
        wind_patterns = [
            r'(\d+)\s*(?:mph|miles per hour)',
            r'(\d+)\s*(?:kph|km/h|kilometers per hour)',
            r'(\d+)\s*knots',
            r'wind\s*(?:speed)?\s*[:\.]?\s*(\d+)'
        ]
        
        for pattern in wind_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                features['wind_speed'] = float(matches[0])
                break
        
        # Precipitation amount
        precip_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:inches|in|")\s*(?:of)?\s*(?:rain|snow|precip)',
            r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters)\s*(?:of)?\s*(?:rain|snow)',
            r'rainfall\s*[:\.]?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in precip_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                features['precipitation'] = float(matches[0])
                break
        
        return features
    
    def extract_keyword_features(self, text: str) -> Dict:
        """Extract keyword presence features"""
        features = {}
        
        for category, keywords in self.weather_keywords.items():
            for keyword in keywords:
                if keyword in text.lower():
                    features[f'has_{category}'] = 1
                    break
            else:
                features[f'has_{category}'] = 0
        
        # Check for severity patterns
        for severity, pattern in self.severity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                features[f'severity_{severity}'] = 1
            else:
                features[f'severity_{severity}'] = 0
        
        return features
    
    def extract_temporal_features(self, timestamp: str) -> Dict:
        """Extract temporal features from timestamp"""
        features = {}
        
        try:
            dt = pd.to_datetime(timestamp)
            
            # Time of day features
            features['hour'] = dt.hour
            features['is_night'] = int(dt.hour < 6 or dt.hour > 18)
            features['is_rush_hour'] = int(7 <= dt.hour <= 9 or 16 <= dt.hour <= 18)
            
            # Day features
            features['day_of_week'] = dt.dayofweek
            features['is_weekend'] = int(dt.dayofweek >= 5)
            features['day_of_month'] = dt.day
            features['month'] = dt.month
            features['quarter'] = (dt.month - 1) // 3 + 1
            
            # Seasonal features
            features['is_winter'] = int(dt.month in [12, 1, 2])
            features['is_spring'] = int(dt.month in [3, 4, 5])
            features['is_summer'] = int(dt.month in [6, 7, 8])
            features['is_fall'] = int(dt.month in [9, 10, 11])
            
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            # Set default values
            features.update({
                'hour': 12,
                'is_night': 0,
                'is_rush_hour': 0,
                'day_of_week': 0,
                'is_weekend': 0,
                'day_of_month': 1,
                'month': 1,
                'quarter': 1,
                'is_winter': 0,
                'is_spring': 0,
                'is_summer': 0,
                'is_fall': 0
            })
        
        return features
    
    def extract_text_statistics(self, text: str) -> Dict:
        """Extract text statistics"""
        features = {}
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['has_exclamation'] = int('!' in text)
        features['has_question'] = int('?' in text)
        features['has_number'] = int(bool(re.search(r'\d', text)))
        features['has_caps'] = int(bool(re.search(r'[A-Z]{2,}', text)))
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe"""
        logger.info(f"Starting dataframe preprocessing. Shape: {df.shape}")
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        # Make a copy to avoid warnings
        df_processed = df.copy()
        
        # Clean text columns
        text_columns = ['title', 'description']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[f'{col}_cleaned'] = df_processed[col].apply(self.clean_text)
        
        # Extract features for each row
        feature_dicts = []
        
        for idx, row in df_processed.iterrows():
            features = {}
            
            # Combine text for analysis
            combined_text = ''
            for col in text_columns:
                if f'{col}_cleaned' in row:
                    combined_text += ' ' + str(row[f'{col}_cleaned'])
            
            # Extract different feature types
            features.update(self.extract_keyword_features(combined_text))
            features.update(self.extract_numerical_features(combined_text))
            features.update(self.extract_text_statistics(combined_text))
            
            # Extract temporal features
            if 'timestamp' in row:
                features.update(self.extract_temporal_features(row['timestamp']))
            
            feature_dicts.append(features)
        
        # Create features dataframe
        if feature_dicts:
            features_df = pd.DataFrame(feature_dicts)
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            # Combine with original data
            df_processed = pd.concat([df_processed, features_df], axis=1)
        
        logger.info(f"Preprocessing complete. New shape: {df_processed.shape}")
        return df_processed
    
    def create_aggregated_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated dataset for ML models"""
        logger.info("Creating aggregated dataset")
        
        if df.empty:
            logger.warning("Empty dataframe, cannot aggregate")
            return pd.DataFrame()
        
        # Ensure we have date column
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
        elif 'date' not in df.columns:
            logger.error("No date column found for aggregation")
            return pd.DataFrame()
        
        # Convert date to datetime for proper grouping
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize aggregated dataframe
        aggregated = pd.DataFrame(index=pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='D'
        ))
        aggregated.index.name = 'date'
        
        # 1. Daily alert counts by type
        if 'alert_type' in df.columns:
            type_counts = df.groupby(['date', 'alert_type']).size().unstack(fill_value=0)
            aggregated = aggregated.join(type_counts)
        
        # 2. Daily severity counts
        severity_cols = [col for col in df.columns if col.startswith('severity_')]
        if severity_cols:
            severity_counts = df.groupby('date')[severity_cols].sum()
            aggregated = aggregated.join(severity_counts)
        
        # 3. Daily keyword feature sums
        keyword_cols = [col for col in df.columns if col.startswith('has_')]
        if keyword_cols:
            keyword_sums = df.groupby('date')[keyword_cols].sum()
            aggregated = aggregated.join(keyword_sums)
        
        # 4. Daily numerical feature averages
        numerical_cols = ['temperature', 'wind_speed', 'precipitation', 
                         'text_length', 'word_count', 'avg_word_length']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if numerical_cols:
            numerical_avg = df.groupby('date')[numerical_cols].mean()
            aggregated = aggregated.join(numerical_avg)
        
        # 5. Total alerts per day
        aggregated['total_alerts'] = df.groupby('date').size()
        
        # Fill NaN values
        aggregated = aggregated.fillna(0)
        
        # Add date features
        aggregated['day_of_week'] = aggregated.index.dayofweek
        aggregated['month'] = aggregated.index.month
        aggregated['day_of_year'] = aggregated.index.dayofyear
        aggregated['week_of_year'] = aggregated.index.isocalendar().week
        aggregated['is_weekend'] = (aggregated['day_of_week'] >= 5).astype(int)
        
        logger.info(f"Aggregated dataset created. Shape: {aggregated.shape}")
        return aggregated

def run_preprocessing_pipeline(raw_data_path: str = None, 
                               processed_data_path: str = None,
                               aggregated_data_path: str = None):
    """Run complete preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)
    
    if raw_data_path is None:
        raw_data_path = Config.RAW_DATA_PATH
    if processed_data_path is None:
        processed_data_path = Config.PROCESSED_DATA_PATH
    if aggregated_data_path is None:
        aggregated_data_path = Config.AGGREGATED_DATA_PATH
    
    try:
        # Load raw data
        if not os.path.exists(raw_data_path):
            logger.error(f"Raw data file not found: {raw_data_path}")
            return None, None
        
        df_raw = pd.read_csv(raw_data_path)
        logger.info(f"Loaded raw data: {df_raw.shape}")
        
        if df_raw.empty:
            logger.warning("Raw data is empty")
            return None, None
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Preprocess data
        df_processed = preprocessor.preprocess_dataframe(df_raw)
        
        # Create aggregated dataset
        df_aggregated = preprocessor.create_aggregated_dataset(df_processed)
        
        # Save processed data
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df_processed.to_csv(processed_data_path, index=False)
        logger.info(f"Saved processed data to {processed_data_path}")
        
        # Save aggregated data
        os.makedirs(os.path.dirname(aggregated_data_path), exist_ok=True)
        df_aggregated.to_csv(aggregated_data_path)
        logger.info(f"Saved aggregated data to {aggregated_data_path}")
        
        return df_processed, df_aggregated
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run preprocessing pipeline
    df_proc, df_agg = run_preprocessing_pipeline(
        Config.RAW_DATA_PATH,
        Config.PROCESSED_DATA_PATH,
        Config.AGGREGATED_DATA_PATH
    )
    
    if df_proc is not None and df_agg is not None:
        print(f"Preprocessing complete. Processed: {df_proc.shape}, Aggregated: {df_agg.shape}")
    else:
        print("Preprocessing failed")
