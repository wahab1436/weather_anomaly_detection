import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Simplified anomaly detector."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in alert data."""
        if df.empty or len(df) < 5:
            logger.warning("Insufficient data for anomaly detection")
            return pd.DataFrame()
        
        # Ensure we have numerical data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numerical columns for anomaly detection")
            return pd.DataFrame()
        
        # Use total alerts or first numeric column
        target_col = 'total_alerts' if 'total_alerts' in df.columns else numeric_cols[0]
        
        # Prepare features
        features = df[[target_col]].copy()
        
        # Add rolling statistics if we have enough data
        if len(features) > 7:
            features['rolling_mean_7'] = features[target_col].rolling(7, min_periods=1).mean()
            features['rolling_std_7'] = features[target_col].rolling(7, min_periods=1).std()
        
        # Fill NaN
        features = features.fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit and predict
        self.model.fit(scaled_features)
        predictions = self.model.predict(scaled_features)
        
        # Create results
        result_df = df.copy()
        result_df['is_anomaly'] = [1 if p == -1 else 0 for p in predictions]
        result_df['anomaly_score'] = self.model.decision_function(scaled_features)
        
        # Simple anomaly reason
        result_df['anomaly_reason'] = result_df.apply(
            lambda row: self._simple_explanation(row, df, target_col), axis=1
        )
        
        return result_df
    
    def _simple_explanation(self, row, df, target_col):
        """Generate simple explanation for anomaly."""
        if row['is_anomaly'] == 0:
            return "Normal"
        
        current = row[target_col]
        if len(df) > 7:
            avg = df[target_col].tail(7).mean()
            if current > avg * 1.5:
                return f"High alert count ({current} vs average {avg:.1f})"
            elif current < avg * 0.5:
                return f"Low alert count ({current} vs average {avg:.1f})"
        
        return "Unusual pattern detected"

def main():
    """Main anomaly detection function."""
    detector = AnomalyDetector()
    
    # Load processed data
    processed_path = 'data/processed/weather_alerts_processed.csv'
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found: {processed_path}")
        return
    
    processed_df = pd.read_csv(processed_path)
    
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
        
        logger.info(f"Anomaly detection completed. Found {anomaly_df['is_anomaly'].sum()} anomalies.")

if __name__ == "__main__":
    main()
