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
        if df.empty:
            return pd.DataFrame(), []
        
        # Make a copy
        features_df = df.copy()
        
        # Select numerical features
        numerical_features = [
            'total_alerts',
            'severity_score',
            'sentiment_score',
            'word_count',
            'alert_intensity',
            '7_day_avg',
            '30_day_avg',
            'day_over_day_change'
        ]
        
        # Add alert type counts
        alert_types = [col for col in df.columns if col in [
            'flood', 'storm', 'wind', 'winter', 'fire', 
            'heat', 'cold', 'coastal', 'air', 'other'
        ]]
        
        numerical_features.extend(alert_types)
        
        # Keep only features that exist in dataframe
        existing_features = [f for f in numerical_features if f in features_df.columns]
        
        # Fill NaN values
        features_df[existing_features] = features_df[existing_features].fillna(0)
        
        # Add temporal features
        if 'issued_date' in features_df.columns:
            features_df['day_of_week'] = pd.to_datetime(features_df['issued_date']).dt.dayofweek
            features_df['day_of_year'] = pd.to_datetime(features_df['issued_date']).dt.dayofyear
            features_df['week_of_year'] = pd.to_datetime(features_df['issued_date']).dt.isocalendar().week
            features_df['month'] = pd.to_datetime(features_df['issued_date']).dt.month
            
            existing_features.extend(['day_of_week', 'day_of_year', 'week_of_year', 'month'])
        
        # Add lag features (previous day values)
        for lag in [1, 2, 3, 7]:
            lag_col = f'total_alerts_lag_{lag}'
            features_df[lag_col] = features_df['total_alerts'].shift(lag)
            existing_features.append(lag_col)
        
        # Fill lag feature NaN values
        features_df[existing_features] = features_df[existing_features].fillna(0)
        
        # Add rolling statistics
        features_df['rolling_std_7'] = features_df['total_alerts'].rolling(window=7).std()
        features_df['rolling_std_30'] = features_df['total_alerts'].rolling(window=30).std()
        
        # Calculate z-scores
        features_df['z_score_7'] = (features_df['total_alerts'] - features_df['7_day_avg']) / features_df['rolling_std_7'].replace(0, 1)
        features_df['z_score_30'] = (features_df['total_alerts'] - features_df['30_day_avg']) / features_df['rolling_std_30'].replace(0, 1)
        
        existing_features.extend(['rolling_std_7', 'rolling_std_30', 'z_score_7', 'z_score_30'])
        
        # Remove any remaining NaN values
        features_df = features_df.dropna(subset=existing_features)
        
        self.feature_columns = existing_features
        return features_df[existing_features], existing_features
    
    def fit(self, df: pd.DataFrame):
        """Fit the anomaly detection model."""
        try:
            # Prepare features
            features_df, feature_cols = self.prepare_features(df)
            
            if len(features_df) < 10:
                logger.warning("Insufficient data for fitting model")
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
            logger.info(f"PCA explained variance ratio: {sum(self.pca.explained_variance_ratio_):.2%}")
            
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}")
            self.is_fitted = False
            raise
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies in the data."""
        if not self.is_fitted:
            logger.error("Model not fitted. Call fit() first.")
            return df
        
        # Prepare features
        features_df, _ = self.prepare_features(df)
        
        if features_df.empty:
            df['anomaly_score'] = 0
            df['is_anomaly'] = False
            return df
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Apply PCA
        pca_features = self.pca.transform(scaled_features)
        
        # Predict anomalies
        predictions = self.model.predict(pca_features)
        scores = self.model.decision_function(pca_features)
        
        # Convert predictions (1 = normal, -1 = anomaly)
        df_aligned = df.iloc[-len(predictions):].copy()
        df_aligned['anomaly_score'] = scores
        df_aligned['is_anomaly'] = predictions == -1
        
        # Calculate anomaly confidence
        df_aligned['anomaly_confidence'] = np.abs(scores)
        
        # Classify anomaly severity
        df_aligned['anomaly_severity'] = pd.cut(
            df_aligned['anomaly_confidence'],
            bins=[-np.inf, 0.1, 0.3, 0.5, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        # Merge back with original dataframe
        result_df = df.copy()
        for col in ['anomaly_score', 'is_anomaly', 'anomaly_confidence', 'anomaly_severity']:
            result_df[col] = np.nan
            result_df.iloc[-len(df_aligned):, result_df.columns.get_loc(col)] = df_aligned[col].values
        
        return result_df
    
    def explain_anomalies(self, df: pd.DataFrame) -> Dict:
        """Generate explanations for detected anomalies."""
        if 'is_anomaly' not in df.columns:
            return {}
        
        anomalies = df[df['is_anomaly'] == True]
        
        if anomalies.empty:
            return {"message": "No anomalies detected"}
        
        explanations = {}
        
        for idx, row in anomalies.iterrows():
            date_str = row['issued_date'].strftime('%Y-%m-%d') if isinstance(row['issued_date'], pd.Timestamp) else str(row['issued_date'])
            
            explanation = {
                'date': date_str,
                'total_alerts': int(row.get('total_alerts', 0)),
                'severity': row.get('anomaly_severity', 'unknown'),
                'confidence': float(row.get('anomaly_confidence', 0)),
                'reasons': []
            }
            
            # Compare with historical averages
            if '7_day_avg' in row and not pd.isna(row['7_day_avg']):
                deviation = ((row['total_alerts'] - row['7_day_avg']) / row['7_day_avg']) * 100
                if abs(deviation) > 50:
                    explanation['reasons'].append(
                        f"{'Above' if deviation > 0 else 'Below'} 7-day average by {abs(deviation):.1f}%"
                    )
            
            # Check specific alert types
            alert_type_cols = [col for col in ['flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold'] 
                             if col in row and row[col] > 0]
            
            if alert_type_cols:
                main_type = max(alert_type_cols, key=lambda x: row[x])
                explanation['reasons'].append(f"High number of {main_type} alerts")
            
            # Check severity score
            if 'severity_score' in row and row['severity_score'] > 0.7:
                explanation['reasons'].append("High average alert severity")
            
            explanations[date_str] = explanation
        
        return explanations
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            logger.warning("Model not fitted. Nothing to save.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.feature_columns = model_data['feature_columns']
            self.contamination = model_data['contamination']
            self.random_state = model_data['random_state']
            self.is_fitted = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_fitted = False

def run_anomaly_detection(input_path: str, output_path: str, model_path: str = None):
    """Complete anomaly detection pipeline."""
    try:
        # Load processed data
        df = pd.read_csv(input_path)
        df['issued_date'] = pd.to_datetime(df['issued_date'])
        
        logger.info(f"Loaded {len(df)} days of data for anomaly detection")
        
        # Initialize detector
        detector = AnomalyDetector(contamination=0.1)
        
        # Load or train model
        if model_path and os.path.exists(model_path):
            detector.load_model(model_path)
        else:
            detector.fit(df)
            if model_path:
                detector.save_model(model_path)
        
        # Detect anomalies
        result_df = detector.predict(df)
        
        # Generate explanations
        explanations = detector.explain_anomalies(result_df)
        
        # Save results
        result_df.to_csv(output_path, index=False)
        
        # Save explanations as JSON
        explanations_path = output_path.replace('.csv', '_explanations.json')
        with open(explanations_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info(f"Anomaly detection complete. Found {result_df['is_anomaly'].sum()} anomalies")
        logger.info(f"Results saved to {output_path}")
        
        return result_df, explanations
        
    except Exception as e:
        logger.error(f"Anomaly detection pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/weather_alerts_daily.csv"
    output_file = "data/output/anomaly_results.csv"
    model_file = "models/isolation_forest.pkl"
    
    run_anomaly_detection(input_file, output_file, model_file)
