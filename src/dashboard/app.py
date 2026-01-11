src/ml/anomaly_detection.py


"""
Anomaly detection module for weather alerts.
Uses Isolation Forest to detect unusual patterns.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, List
import warnings
import os
import json
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detect anomalies in weather alert patterns."""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for anomaly detection."""
        if df.empty or len(df) < 2:
            return pd.DataFrame(), []
        
        # Make a copy
        features_df = df.copy()
        
        # Select numerical features that might exist
        possible_features = [
            'total_alerts',
            'severity_score',
            'sentiment_score',
            'word_count',
            'alert_intensity',
            '7_day_avg',
            '30_day_avg',
            'day_over_day_change'
        ]
        
        # Keep only features that exist in dataframe
        existing_features = [f for f in possible_features if f in features_df.columns]
        
        # Add alert type counts if they exist
        alert_types = [col for col in df.columns if col in [
            'flood', 'storm', 'wind', 'winter', 'fire', 
            'heat', 'cold', 'coastal', 'air', 'other'
        ]]
        
        existing_features.extend(alert_types)
        
        # If no features found, create some basic ones
        if not existing_features and 'issued_date' in features_df.columns:
            # Create basic temporal features
            features_df['day_of_week'] = pd.to_datetime(features_df['issued_date']).dt.dayofweek
            features_df['day_of_year'] = pd.to_datetime(features_df['issued_date']).dt.dayofyear
            existing_features = ['day_of_week', 'day_of_year']
        
        # Fill NaN values
        if existing_features:
            features_df[existing_features] = features_df[existing_features].fillna(0)
        
        # Add temporal features if date column exists
        if 'issued_date' in features_df.columns:
            if 'day_of_week' not in existing_features:
                features_df['day_of_week'] = pd.to_datetime(features_df['issued_date']).dt.dayofweek
                existing_features.append('day_of_week')
            
            if 'day_of_year' not in existing_features:
                features_df['day_of_year'] = pd.to_datetime(features_df['issued_date']).dt.dayofyear
                existing_features.append('day_of_year')
        
        # Add lag features only if we have enough data
        if 'total_alerts' in features_df.columns and len(features_df) > 7:
            for lag in [1, 2, 3, 7]:
                lag_col = f'total_alerts_lag_{lag}'
                features_df[lag_col] = features_df['total_alerts'].shift(lag)
                existing_features.append(lag_col)
        
        # Add rolling statistics only if we have enough data
        if 'total_alerts' in features_df.columns and len(features_df) > 7:
            features_df['rolling_std_7'] = features_df['total_alerts'].rolling(window=7, min_periods=1).std()
            features_df['rolling_std_30'] = features_df['total_alerts'].rolling(window=30, min_periods=1).std()
            
            # Calculate z-scores
            features_df['z_score_7'] = (features_df['total_alerts'] - features_df.get('7_day_avg', 0)) / features_df['rolling_std_7'].replace(0, 1)
            features_df['z_score_30'] = (features_df['total_alerts'] - features_df.get('30_day_avg', 0)) / features_df['rolling_std_30'].replace(0, 1)
            
            existing_features.extend(['rolling_std_7', 'rolling_std_30', 'z_score_7', 'z_score_30'])
        
        # Remove any remaining NaN values
        if existing_features:
            features_df = features_df.dropna(subset=existing_features)
        
        self.feature_columns = existing_features
        return features_df[existing_features] if existing_features else pd.DataFrame(), existing_features
    
    def fit(self, df: pd.DataFrame):
        """Fit the anomaly detection model."""
        try:
            # Prepare features
            features_df, feature_cols = self.prepare_features(df)
            
            if len(features_df) < 5:  # Reduced minimum threshold
                logger.warning(f"Insufficient data for fitting model. Have {len(features_df)} samples, need at least 5.")
                self.is_fitted = False
                return
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_df)
            
            # Apply PCA for dimensionality reduction
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Fit Isolation Forest
            self.model.fit(pca_features)
            self.is_fitted = True
            
            logger.info(f"Anomaly detector fitted on {len(features_df)} samples with {len(feature_cols)} features")
            if hasattr(self.pca, 'explained_variance_ratio_'):
                logger.info(f"PCA explained variance ratio: {sum(self.pca.explained_variance_ratio_):.2%}")
            
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}")
            self.is_fitted = False
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies in the data."""
        # Always return a result, even if model isn't fitted
        result_df = df.copy()
        
        if not self.is_fitted:
            logger.warning("Model not fitted. Returning default (no anomalies).")
            result_df['anomaly_score'] = 0
            result_df['is_anomaly'] = False
            result_df['anomaly_confidence'] = 0
            result_df['anomaly_severity'] = 'low'
            return result_df
        
        try:
            # Prepare features
            features_df, _ = self.prepare_features(df)
            
            if features_df.empty:
                result_df['anomaly_score'] = 0
                result_df['is_anomaly'] = False
                result_df['anomaly_confidence'] = 0
                result_df['anomaly_severity'] = 'low'
                return result_df
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Apply PCA
            pca_features = self.pca.transform(scaled_features)
            
            # Predict anomalies
            predictions = self.model.predict(pca_features)
            scores = self.model.decision_function(pca_features)
            
            # Convert predictions (1 = normal, -1 = anomaly)
            is_anomaly = predictions == -1
            
            # Add results to dataframe
            result_df = df.copy()
            
            # Initialize with default values
            result_df['anomaly_score'] = 0
            result_df['is_anomaly'] = False
            result_df['anomaly_confidence'] = 0
            result_df['anomaly_severity'] = 'low'
            
            # Fill in actual predictions for rows that have them
            if len(is_anomaly) > 0:
                # Align predictions with dataframe
                start_idx = len(result_df) - len(is_anomaly)
                if start_idx >= 0:
                    result_df.iloc[start_idx:, result_df.columns.get_loc('anomaly_score')] = scores
                    result_df.iloc[start_idx:, result_df.columns.get_loc('is_anomaly')] = is_anomaly
                    result_df.iloc[start_idx:, result_df.columns.get_loc('anomaly_confidence')] = np.abs(scores)
                    
                    # Classify anomaly severity
                    confidence_values = result_df['anomaly_confidence'].copy()
                    severity = pd.cut(
                        confidence_values,
                        bins=[-np.inf, 0.1, 0.3, 0.5, np.inf],
                        labels=['low', 'medium', 'high', 'critical'],
                        include_lowest=True
                    )
                    result_df['anomaly_severity'] = severity
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}. Returning default results.")
            result_df['anomaly_score'] = 0
            result_df['is_anomaly'] = False
            result_df['anomaly_confidence'] = 0
            result_df['anomaly_severity'] = 'low'
            return result_df
    
    def explain_anomalies(self, df: pd.DataFrame) -> Dict:
        """Generate explanations for detected anomalies."""
        explanations = {"message": "No anomalies detected", "anomalies": {}}
        
        if df.empty or 'is_anomaly' not in df.columns:
            return explanations
        
        anomalies = df[df['is_anomaly'] == True]
        
        if anomalies.empty:
            return explanations
        
        explanations["message"] = f"Found {len(anomalies)} anomalies"
        
        for idx, row in anomalies.iterrows():
            # Safely get date
            try:
                if isinstance(row.get('issued_date'), pd.Timestamp):
                    date_str = row['issued_date'].strftime('%Y-%m-%d')
                elif 'date' in row:
                    date_str = str(row['date'])
                else:
                    date_str = f"row_{idx}"
            except:
                date_str = f"row_{idx}"
            
            explanation = {
                'date': date_str,
                'total_alerts': int(row.get('total_alerts', 0)),
                'severity': str(row.get('anomaly_severity', 'unknown')),
                'confidence': float(row.get('anomaly_confidence', 0)),
                'reasons': []
            }
            
            # Compare with historical averages
            if '7_day_avg' in row and not pd.isna(row['7_day_avg']) and row['7_day_avg'] > 0:
                deviation = ((row.get('total_alerts', 0) - row['7_day_avg']) / row['7_day_avg']) * 100
                if abs(deviation) > 50:
                    explanation['reasons'].append(
                        f"{'Above' if deviation > 0 else 'Below'} 7-day average by {abs(deviation):.1f}%"
                    )
            
            # Check specific alert types
            alert_type_cols = [col for col in ['flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold'] 
                             if col in row and pd.notna(row[col]) and row[col] > 0]
            
            if alert_type_cols:
                # Find the alert type with maximum value
                max_val = 0
                main_type = ''
                for col in alert_type_cols:
                    if row[col] > max_val:
                        max_val = row[col]
                        main_type = col
                if main_type:
                    explanation['reasons'].append(f"High number of {main_type} alerts")
            
            # Check severity score
            if 'severity_score' in row and pd.notna(row['severity_score']) and row['severity_score'] > 0.7:
                explanation['reasons'].append("High average alert severity")
            
            explanations["anomalies"][date_str] = explanation
        
        return explanations
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            logger.warning("Model not fitted. Nothing to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.feature_columns = model_data.get('feature_columns', [])
            self.contamination = model_data.get('contamination', 0.1)
            self.random_state = model_data.get('random_state', 42)
            self.is_fitted = model_data.get('is_fitted', True)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_fitted = False

def run_anomaly_detection(input_path: str, output_path: str, model_path: str = None):
    """Complete anomaly detection pipeline."""
    try:
        # Load processed data
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            # Create a simple default result
            result_df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'total_alerts': [0],
                'anomaly_score': [0],
                'is_anomaly': [False],
                'anomaly_confidence': [0],
                'anomaly_severity': ['low']
            })
            result_df.to_csv(output_path, index=False)
            return result_df, {"message": "No data available"}
        
        df = pd.read_csv(input_path)
        
        # Check if dataframe is empty
        if df.empty:
            logger.warning("Input dataframe is empty")
            df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'total_alerts': [0]
            })
        
        # Convert date column
        date_column = None
        for col in ['issued_date', 'date', 'timestamp']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            # Fill NaT with current date
            df[date_column] = df[date_column].fillna(pd.Timestamp.now())
        
        logger.info(f"Loaded {len(df)} days of data for anomaly detection")
        
        # Initialize detector
        detector = AnomalyDetector(contamination=0.1)
        
        # Load or train model
        model_loaded = False
        if model_path and os.path.exists(model_path):
            try:
                detector.load_model(model_path)
                model_loaded = detector.is_fitted
                if model_loaded:
                    logger.info("Successfully loaded pre-trained model")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        
        # If model not loaded and we have enough data, try to train
        if not detector.is_fitted and len(df) >= 5:
            logger.info("Training new anomaly detection model...")
            try:
                detector.fit(df)
                if detector.is_fitted and model_path:
                    detector.save_model(model_path)
                    logger.info(f"Model trained and saved to {model_path}")
            except Exception as e:
                logger.warning(f"Could not train model: {e}")
        
        # Detect anomalies (this will work even if model isn't fitted)
        result_df = detector.predict(df)
        
        # Ensure required columns exist
        required_columns = ['anomaly_score', 'is_anomaly', 'anomaly_confidence', 'anomaly_severity']
        for col in required_columns:
            if col not in result_df.columns:
                result_df[col] = 0 if col != 'anomaly_severity' else 'low'
                if col == 'is_anomaly':
                    result_df[col] = False
        
        # Generate explanations
        explanations = detector.explain_anomalies(result_df)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        result_df.to_csv(output_path, index=False)
        
        # Save explanations as JSON if we have any
        if explanations and explanations.get("anomalies"):
            explanations_path = output_path.replace('.csv', '_explanations.json')
            with open(explanations_path, 'w') as f:
                json.dump(explanations, f, indent=2)
            logger.info(f"Explanations saved to {explanations_path}")
        
        # Safely count anomalies
        anomaly_count = 0
        if 'is_anomaly' in result_df.columns:
            anomaly_count = result_df['is_anomaly'].sum()
        
        logger.info(f"Anomaly detection complete. Found {anomaly_count} anomalies")
        logger.info(f"Results saved to {output_path}")
        
        return result_df, explanations
        
    except Exception as e:
        logger.error(f"Anomaly detection pipeline failed: {str(e)}")
        # Create a default result to prevent complete failure
        try:
            result_df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'total_alerts': [0],
                'anomaly_score': [0],
                'is_anomaly': [False],
                'anomaly_confidence': [0],
                'anomaly_severity': ['low']
            })
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            return result_df, {"message": f"Pipeline failed: {str(e)}"}
        except:
            # Last resort
            return pd.DataFrame(), {"message": "Complete pipeline failure"}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    input_file = "data/processed/weather_alerts_daily.csv"
    output_file = "data/output/anomaly_results.csv"
    model_file = "models/isolation_forest.pkl"
    
    # Create directories if they don't exist
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    result_df, explanations = run_anomaly_detection(input_file, output_file, model_file)
    print(f"Processed {len(result_df)} records")
    print(f"Found {result_df['is_anomaly'].sum()} anomalies")

