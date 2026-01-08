"""
Anomaly detection module for weather alerts
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import logging
from typing import Tuple, Dict, Any, List
import joblib
from datetime import datetime, timedelta
import warnings
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detect anomalies in weather alert patterns"""
    
    def __init__(self, contamination: float = None):
        self.contamination = contamination or Config.ANOMALY_CONTAMINATION
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_columns = []
        self.is_fitted = False
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for anomaly detection"""
        # Exclude date columns and target columns
        exclude_patterns = ['date', 'timestamp', 'is_anomaly', 'anomaly_']
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns
        feature_cols = []
        for col in numeric_cols:
            if not any(pattern in col.lower() for pattern in exclude_patterns):
                feature_cols.append(col)
        
        # Prioritize alert count features
        priority_cols = [col for col in feature_cols if 'alert' in col.lower() or 'total' in col.lower()]
        other_cols = [col for col in feature_cols if col not in priority_cols]
        
        # Limit number of features to avoid overfitting
        max_features = min(50, len(priority_cols) + len(other_cols))
        selected_cols = priority_cols + other_cols[:max_features - len(priority_cols)]
        
        logger.info(f"Selected {len(selected_cols)} features for anomaly detection")
        return selected_cols
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare features for anomaly detection"""
        logger.info("Preparing features for anomaly detection")
        
        # Select features
        self.feature_columns = self.select_features(df)
        
        if not self.feature_columns:
            logger.warning("No features selected for anomaly detection")
            return np.array([]), df
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Handle infinite and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        
        logger.info(f"Prepared features shape: {X_scaled.shape}")
        return X_scaled, df[self.feature_columns]
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """Detect statistical anomalies using rolling statistics"""
        result_df = df.copy()
        
        # Focus on alert count columns
        alert_cols = [col for col in result_df.columns 
                     if 'alert' in col.lower() or 'total' in col.lower()]
        
        for col in alert_cols:
            if col in result_df.columns and len(result_df) > window:
                # Calculate rolling statistics
                rolling_mean = result_df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = result_df[col].rolling(window=window, min_periods=1).std()
                
                # Replace zero std with small value
                rolling_std = rolling_std.replace(0, 1e-6)
                
                # Calculate z-score
                z_score = (result_df[col] - rolling_mean) / rolling_std
                
                # Flag anomalies (z-score > 3 or < -3)
                result_df[f'{col}_zscore'] = z_score
                result_df[f'{col}_stat_anomaly'] = np.abs(z_score) > 3
        
        return result_df
    
    def detect_multi_model_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using multiple models for consensus"""
        result_df = df.copy()
        
        # Prepare features
        X_scaled, _ = self.prepare_features(df)
        
        if X_scaled.shape[0] == 0:
            logger.warning("No data for multi-model anomaly detection")
            return result_df
        
        # Train multiple models
        models = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=Config.RANDOM_STATE
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=self.contamination,
                random_state=Config.RANDOM_STATE
            )
        }
        
        anomaly_scores = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'fit_predict'):
                    predictions = model.fit_predict(X_scaled)
                    scores = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else predictions
                else:
                    model.fit(X_scaled)
                    predictions = model.predict(X_scaled)
                    scores = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else predictions
                
                # Store results
                result_df[f'{name}_anomaly'] = predictions == -1
                if scores is not None:
                    result_df[f'{name}_score'] = scores
                    anomaly_scores[name] = scores
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        # Calculate consensus anomaly (majority vote)
        anomaly_cols = [col for col in result_df.columns if col.endswith('_anomaly')]
        if len(anomaly_cols) > 0:
            result_df['consensus_anomaly'] = result_df[anomaly_cols].sum(axis=1) > (len(anomaly_cols) / 2)
        
        # Calculate average anomaly score
        score_cols = [col for col in result_df.columns if col.endswith('_score')]
        if len(score_cols) > 0:
            result_df['avg_anomaly_score'] = result_df[score_cols].mean(axis=1)
        
        return result_df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the dataset using multiple methods"""
        logger.info("Starting anomaly detection")
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            df['is_anomaly'] = False
            df['anomaly_score'] = 0
            df['anomaly_probability'] = 0
            return df
        
        result_df = df.copy()
        
        # Method 1: Statistical anomalies
        result_df = self.detect_statistical_anomalies(result_df)
        
        # Method 2: Isolation Forest
        X_scaled, _ = self.prepare_features(result_df)
        
        if X_scaled.shape[0] > 0:
            # Fit model if not already fitted
            if not hasattr(self.model, 'estimators_'):
                self.model.fit(X_scaled)
            
            # Get anomaly scores and predictions
            anomaly_scores = self.model.decision_function(X_scaled)
            predictions = self.model.predict(X_scaled)
            
            # Store results
            result_df['isolation_forest_anomaly'] = predictions == -1
            result_df['isolation_forest_score'] = anomaly_scores
            
            # Convert to probability (0 to 1, where 1 is most anomalous)
            min_score = anomaly_scores.min()
            max_score = anomaly_scores.max()
            if max_score > min_score:
                result_df['anomaly_probability'] = (
                    (anomaly_scores - min_score) / (max_score - min_score)
                )
            else:
                result_df['anomaly_probability'] = 0.5
            
            # Final anomaly flag (combination of methods)
            stat_anomaly_cols = [col for col in result_df.columns if col.endswith('_stat_anomaly')]
            if stat_anomaly_cols:
                has_stat_anomaly = result_df[stat_anomaly_cols].any(axis=1)
                result_df['is_anomaly'] = (
                    result_df['isolation_forest_anomaly'] | 
                    has_stat_anomaly
                )
            else:
                result_df['is_anomaly'] = result_df['isolation_forest_anomaly']
            
            result_df['anomaly_score'] = anomaly_scores
            
        else:
            logger.warning("Could not compute anomaly scores")
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = 0
            result_df['anomaly_probability'] = 0
        
        # Count anomalies
        anomaly_count = result_df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(result_df)*100:.1f}%)")
        
        return result_df
    
    def explain_anomalies(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Generate explanations for detected anomalies"""
        if df.empty or 'is_anomaly' not in df.columns:
            return pd.DataFrame()
        
        anomalies = df[df['is_anomaly']].copy()
        
        if anomalies.empty:
            return pd.DataFrame()
        
        explanations = []
        
        for idx, row in anomalies.iterrows():
            explanation = {
                'date': row.name if hasattr(row, 'name') else idx,
                'anomaly_score': row.get('anomaly_score', 0),
                'anomaly_probability': row.get('anomaly_probability', 0),
                'reason': []
            }
            
            # Check for high alert counts
            if 'total_alerts' in row:
                if row['total_alerts'] > df['total_alerts'].quantile(0.95):
                    explanation['reason'].append(f"High alert count: {row['total_alerts']}")
            
            # Check for specific alert types
            alert_cols = [col for col in row.index if col in Config.ALERT_TYPES.keys()]
            for col in alert_cols:
                if row[col] > df[col].quantile(0.95):
                    explanation['reason'].append(f"High {col} alerts: {row[col]}")
            
            # Check for statistical anomalies
            stat_cols = [col for col in row.index if col.endswith('_stat_anomaly') and row[col]]
            if stat_cols:
                explanation['reason'].append("Statistical anomaly detected")
            
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = Config.ANOMALY_MODEL_PATH
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'contamination': self.contamination
        }, filepath)
        logger.info(f"Saved anomaly detection model to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = Config.ANOMALY_MODEL_PATH
        
        if os.path.exists(filepath):
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_columns = saved_data['feature_columns']
            self.is_fitted = saved_data['is_fitted']
            self.contamination = saved_data.get('contamination', Config.ANOMALY_CONTAMINATION)
            logger.info(f"Loaded anomaly detection model from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")

def run_anomaly_detection(aggregated_data_path: str = None, 
                         output_path: str = None,
                         model_path: str = None):
    """Run complete anomaly detection pipeline"""
    logger.info("=" * 60)
    logger.info("Starting anomaly detection pipeline")
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)
    
    if aggregated_data_path is None:
        aggregated_data_path = Config.AGGREGATED_DATA_PATH
    if output_path is None:
        output_path = Config.ANOMALY_OUTPUT_PATH
    if model_path is None:
        model_path = Config.ANOMALY_MODEL_PATH
    
    try:
        # Load aggregated data
        if not os.path.exists(aggregated_data_path):
            logger.error(f"Aggregated data file not found: {aggregated_data_path}")
            return None
        
        df = pd.read_csv(aggregated_data_path, index_col=0)
        
        # Handle index
        if df.index.name == 'date':
            df.index = pd.to_datetime(df.index)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
        
        logger.info(f"Loaded aggregated data: {df.shape}")
        
        if df.empty:
            logger.warning("Aggregated data is empty")
            return None
        
        # Initialize detector
        detector = AnomalyDetector()
        
        # Try to load existing model
        detector.load_model(model_path)
        
        # Detect anomalies
        df_anomalies = detector.detect_anomalies(df)
        
        # Generate explanations for top anomalies
        explanations = detector.explain_anomalies(df_anomalies)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_anomalies.to_csv(output_path)
        
        # Save explanations if any
        if not explanations.empty:
            explanations_path = output_path.replace('.csv', '_explanations.csv')
            explanations.to_csv(explanations_path, index=False)
            logger.info(f"Saved anomaly explanations to {explanations_path}")
        
        # Save model
        detector.save_model(model_path)
        
        logger.info(f"Saved anomaly results to {output_path}")
        logger.info(f"Total anomalies detected: {df_anomalies['is_anomaly'].sum()}")
        
        return df_anomalies
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run anomaly detection
    df_anomalies = run_anomaly_detection(
        Config.AGGREGATED_DATA_PATH,
        Config.ANOMALY_OUTPUT_PATH,
        Config.ANOMALY_MODEL_PATH
    )
    
    if df_anomalies is not None:
        anomaly_count = df_anomalies['is_anomaly'].sum()
        print(f"Anomaly detection complete. Detected {anomaly_count} anomalies.")
    else:
        print("Anomaly detection failed")
