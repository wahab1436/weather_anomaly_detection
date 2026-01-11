import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
import os
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detects anomalies in weather alert patterns."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        if df.empty:
            return pd.DataFrame()
        
        features = df.copy()
        
        # Ensure date column exists
        if 'date' in features.columns:
            features = features.set_index('date')
        
        # Select numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create time-based features
        if len(features) > 1:
            # Rolling statistics
            for window in [3, 7, 14]:
                for col in ['total_alerts']:
                    if col in features.columns:
                        features[f'{col}_rolling_mean_{window}'] = features[col].rolling(window=window, min_periods=1).mean()
                        features[f'{col}_rolling_std_{window}'] = features[col].rolling(window=window, min_periods=1).std()
            
            # Percentage changes
            for col in ['total_alerts']:
                if col in features.columns:
                    features[f'{col}_pct_change'] = features[col].pct_change().fillna(0)
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Select final feature columns
        feature_cols = [col for col in features.columns if col in numerical_cols or 'rolling' in col or 'pct_change' in col]
        
        return features[feature_cols]
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in alert data."""
        if df.empty or len(df) < 3:  # Need minimum data points
            logger.warning("Insufficient data for anomaly detection")
            return pd.DataFrame()
        
        # Prepare features
        features = self.prepare_features(df)
        
        if features.empty:
            return pd.DataFrame()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(scaled_features)
        self.is_fitted = True
        
        # Predict anomalies
        predictions = self.model.predict(scaled_features)
        
        # Convert predictions: 1 = normal, -1 = anomaly
        df_result = df.copy()
        df_result['is_anomaly'] = [1 if p == -1 else 0 for p in predictions]
        
        # Calculate anomaly scores
        anomaly_scores = self.model.decision_function(scaled_features)
        df_result['anomaly_score'] = anomaly_scores
        
        # Flag significant anomalies (score < threshold)
        threshold = np.percentile(anomaly_scores, self.contamination * 100)
        df_result['significant_anomaly'] = (anomaly_scores < threshold).astype(int)
        
        # Add reason for anomaly
        df_result['anomaly_reason'] = df_result.apply(
            lambda row: self._explain_anomaly(row, df), axis=1
        )
        
        logger.info(f"Detected {df_result['is_anomaly'].sum()} anomalies")
        
        return df_result
    
    def _explain_anomaly(self, row: pd.Series, df: pd.DataFrame) -> str:
        """Generate plain English explanation for anomaly."""
        if row['is_anomaly'] == 0:
            return "Normal"
        
        reasons = []
        
        # Check for spike in total alerts
        if 'total_alerts' in row.index:
            current = row['total_alerts']
            if len(df) > 7:
                avg_last_week = df['total_alerts'].iloc[-7:].mean()
                if current > avg_last_week * 2:
                    reasons.append(f"Alert count ({current}) is double the recent average ({avg_last_week:.1f})")
        
        # Check specific alert types
        alert_type_cols = [col for col in df.columns if 'alert_type' in str(col)]
        for col in alert_type_cols:
            if col in row.index and row[col] > 0:
                if len(df) > 7:
                    avg_type = df[col].iloc[-7:].mean()
                    if row[col] > avg_type * 3:
                        alert_type = col.replace('alert_type_', '')
                        reasons.append(f"{alert_type} alerts spiked ({row[col]} vs average {avg_type:.1f})")
        
        if reasons:
            return "; ".join(reasons)
        return "Unusual pattern detected"
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        if self.is_fitted:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'contamination': self.contamination
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.contamination = model_data.get('contamination', 0.1)
            self.is_fitted = True
            logger.info(f"Loaded model from {filepath}")
    
    def generate_insights(self, anomaly_df: pd.DataFrame) -> Dict:
        """Generate plain English insights from anomalies."""
        insights = {
            'summary': '',
            'key_anomalies': [],
            'recommendations': []
        }
        
        if anomaly_df.empty:
            insights['summary'] = "No recent anomalies detected. Weather patterns are within normal ranges."
            return insights
        
        # Get recent anomalies
        recent_anomalies = anomaly_df[anomaly_df['is_anomaly'] == 1].tail(5)
        
        if recent_anomalies.empty:
            insights['summary'] = "No significant anomalies in the past week."
            return insights
        
        # Generate summary
        total_anomalies = len(recent_anomalies)
        insights['summary'] = f"Detected {total_anomalies} unusual weather patterns in recent data."
        
        # List key anomalies
        for idx, row in recent_anomalies.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if 'date' in row else 'Recent'
            alert_count = row.get('total_alerts', 0)
            
            anomaly_info = {
                'date': date_str,
                'alert_count': alert_count,
                'reason': row.get('anomaly_reason', 'Unusual pattern'),
                'score': round(row.get('anomaly_score', 0), 3)
            }
            insights['key_anomalies'].append(anomaly_info)
        
        # Generate recommendations
        if total_anomalies > 3:
            insights['recommendations'].append("Multiple anomalies detected. Consider reviewing regional weather patterns.")
        
        max_anomaly = recent_anomalies['total_alerts'].max()
        if max_anomaly > 20:
            insights['recommendations'].append(f"High alert volume ({max_anomaly}) detected. Monitor for developing severe weather.")
        
        return insights

def main():
    """Main anomaly detection function."""
    detector = AnomalyDetector()
    
    # Load processed data
    processed_path = 'data/processed/weather_alerts_processed.csv'
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found: {processed_path}")
        return
    
    processed_df = pd.read_csv(processed_path, parse_dates=['date'])
    
    if processed_df.empty:
        logger.warning("No data for anomaly detection")
        return
    
    # Detect anomalies
    anomaly_df = detector.detect_anomalies(processed_df)
    
    if not anomaly_df.empty:
        # Save results
        output_path = 'data/output/anomaly_detection_results.csv'
        os.makedirs('data/output', exist_ok=True)
        anomaly_df.to_csv(output_path, index=False)
        
        # Generate insights
        insights = detector.generate_insights(anomaly_df)
        
        # Save insights
        insights_path = 'data/output/anomaly_insights.json'
        import json
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Save model
        model_path = 'models/isolation_forest.pkl'
        detector.save_model(model_path)
        
        logger.info("Anomaly detection completed successfully")
        
        # Print summary
        print(f"Anomalies detected: {anomaly_df['is_anomaly'].sum()}")
        print(f"Insights saved to: {insights_path}")

if __name__ == "__main__":
    main()
