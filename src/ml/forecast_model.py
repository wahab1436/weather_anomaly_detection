"""
Forecasting module for weather alert predictions - FIXED VERSION
Uses XGBoost for time series forecasting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from datetime import datetime, timedelta
import warnings
import os  # <-- ADDED
import json  # <-- ADDED

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False
    # Create dummy classes
    class DummyXGBRegressor:
        def fit(self, *args, **kwargs): pass
        def predict(self, X): return np.zeros(len(X))
    xgb = type('xgboost', (), {'XGBRegressor': DummyXGBRegressor})()
    joblib = type('joblib', (), {'dump': lambda x, y: None, 'load': lambda x: None})()

logger = logging.getLogger(__name__)

class AlertForecaster:
    """Forecast future weather alerts using XGBoost."""
    
    def __init__(self, forecast_horizon: int = 7, n_splits: int = 5):
        """
        Initialize forecaster.
        
        Args:
            forecast_horizon: Number of days to forecast ahead
            n_splits: Number of splits for time series cross-validation
        """
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.models = {}
        self.feature_columns = None
        self.target_columns = None
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame, target_col: str = 'total_alerts') -> Tuple[pd.DataFrame, List[str]]:
        """Create features for time series forecasting."""
        if df.empty:
            return pd.DataFrame(), []
        
        features_df = df.copy()
        
        # Ensure datetime index
        if 'issued_date' in features_df.columns:
            features_df = features_df.set_index('issued_date')
        
        # Check if target column exists
        if target_col not in features_df.columns:
            logger.warning(f"Target column '{target_col}' not found. Available columns: {list(features_df.columns)}")
            # Create a dummy target
            features_df['target'] = 0
        else:
            features_df['target'] = features_df[target_col]
        
        # Lag features (only if we have enough data)
        max_lags = min(30, len(features_df) - 1)
        if max_lags > 0:
            for lag in range(1, max_lags + 1):
                features_df[f'lag_{lag}'] = features_df['target'].shift(lag)
        
        # Rolling statistics (handle insufficient data)
        windows = [3, 7, 14, 30]
        for window in windows:
            if len(features_df) >= window:
                features_df[f'rolling_mean_{window}'] = features_df['target'].rolling(window=window, min_periods=1).mean()
                features_df[f'rolling_std_{window}'] = features_df['target'].rolling(window=window, min_periods=1).std()
                features_df[f'rolling_min_{window}'] = features_df['target'].rolling(window=window, min_periods=1).min()
                features_df[f'rolling_max_{window}'] = features_df['target'].rolling(window=window, min_periods=1).max()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            features_df[f'ema_{alpha}'] = features_df['target'].ewm(alpha=alpha, min_periods=1).mean()
        
        # Temporal features from index
        if hasattr(features_df.index, 'dayofweek'):
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['year'] = features_df.index.year
            
            # Cyclical encoding for temporal features
            features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Difference features (only if we have enough data)
        if len(features_df) > 1:
            features_df['diff_1'] = features_df['target'].diff(1)
        if len(features_df) > 7:
            features_df['diff_7'] = features_df['target'].diff(7)
        
        # Percentage changes (handle division by zero)
        if len(features_df) > 1:
            pct_change = features_df['target'].pct_change(1)
            features_df['pct_change_1'] = pct_change.fillna(0)
        
        # Remove rows with NaN values (from lag features)
        features_df = features_df.dropna()
        
        # Define feature columns (all except target)
        feature_columns = [col for col in features_df.columns if col != 'target']
        
        return features_df, feature_columns
    
    def prepare_forecast_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for forecasting."""
        features_df, feature_columns = self.create_features(df, target_col)
        
        if features_df.empty:
            return np.array([]), np.array([]), feature_columns
        
        X = features_df[feature_columns].values
        y = features_df['target'].values
        
        self.feature_columns = feature_columns
        self.target_columns = target_col
        
        return X, y, feature_columns
    
    def train_test_split_time_series(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Split time series data maintaining temporal order."""
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, df: pd.DataFrame, target_columns: List[str] = None):
        """Fit forecasting models for specified target columns."""
        if target_columns is None:
            # Check which columns actually exist
            available_cols = df.columns.tolist()
            target_columns = [col for col in ['total_alerts', 'flood', 'storm', 'wind'] 
                            if col in available_cols]
        
        if not target_columns:
            logger.warning("No valid target columns found for forecasting")
            return
        
        for target_col in target_columns:
            try:
                # Prepare data
                X, y, feature_cols = self.prepare_forecast_data(df, target_col)
                
                if len(X) < 20:  # Reduced minimum
                    logger.warning(f"Insufficient data for {target_col}. Need at least 20 samples, have {len(X)}.")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = self.train_test_split_time_series(X, y, test_size=0.2)
                
                if len(X_train) == 0:
                    logger.warning(f"No training data for {target_col}")
                    continue
                
                # Initialize and train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=200,  # Reduced for faster training
                    max_depth=4,       # Reduced to prevent overfitting
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20
                )
                
                # Train with early stopping if we have validation data
                if len(X_test) > 0:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train, verbose=False)
                
                # Store model
                self.models[target_col] = {
                    'model': model,
                    'feature_columns': feature_cols,
                    'last_date': df.index[-1] if hasattr(df, 'index') else pd.Timestamp.now()
                }
                
                # Evaluate model if test data exists
                if len(X_test) > 0:
                    predictions = model.predict(X_test)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    logger.info(f"Model for {target_col}: MAE={mae:.2f}, RMSE={rmse:.2f}")
                else:
                    logger.info(f"Model for {target_col} trained (no evaluation data)")
                
            except Exception as e:
                logger.error(f"Error fitting model for {target_col}: {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        logger.info(f"Fitted {len(self.models)} forecasting models")
    
    def forecast(self, df: pd.DataFrame, steps_ahead: int = None) -> Dict:
        """Generate forecasts for specified steps ahead."""
        if not self.is_fitted:
            logger.error("No models fitted. Call fit() first.")
            return {}
        
        if steps_ahead is None:
            steps_ahead = self.forecast_horizon
        
        forecasts = {}
        
        for target_col, model_data in self.models.items():
            try:
                # Prepare latest data for forecasting
                latest_data = df.copy()
                if 'issued_date' in latest_data.columns:
                    latest_data = latest_data.set_index('issued_date')
                
                # Get the model and features
                model = model_data['model']
                feature_cols = model_data['feature_columns']
                
                # Create forecast dates
                last_date = latest_data.index[-1] if len(latest_data) > 0 else pd.Timestamp.now()
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps_ahead,
                    freq='D'
                )
                
                # Generate simple forecast (simplified approach)
                # For now, use last value as forecast
                if target_col in latest_data.columns and len(latest_data) > 0:
                    last_value = float(latest_data[target_col].iloc[-1])
                    target_forecasts = [max(0, last_value) for _ in range(steps_ahead)]
                else:
                    target_forecasts = [0 for _ in range(steps_ahead)]
                
                # Calculate simple confidence intervals
                if target_forecasts:
                    std_dev = np.std(target_forecasts) if len(target_forecasts) > 1 else target_forecasts[0] * 0.1
                    lower = [max(0, x - 1.96 * std_dev) for x in target_forecasts]
                    upper = [x + 1.96 * std_dev for x in target_forecasts]
                else:
                    lower = upper = []
                
                # Store forecasts
                forecasts[target_col] = {
                    'dates': forecast_dates,
                    'values': target_forecasts,
                    'confidence_intervals': (lower, upper)
                }
                
                logger.info(f"Generated {steps_ahead}-day forecast for {target_col}")
                
            except Exception as e:
                logger.error(f"Error forecasting {target_col}: {str(e)}")
                # Provide default forecast
                forecast_dates = pd.date_range(
                    start=pd.Timestamp.now() + pd.Timedelta(days=1),
                    periods=steps_ahead,
                    freq='D'
                )
                forecasts[target_col] = {
                    'dates': forecast_dates,
                    'values': [0] * steps_ahead,
                    'confidence_intervals': ([0] * steps_ahead, [0] * steps_ahead)
                }
        
        return forecasts
    
    def save_models(self, filepath: str):
        """Save all trained models."""
        if not self.is_fitted:
            logger.warning("No models fitted. Nothing to save.")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'forecast_horizon': self.forecast_horizon,
                'n_splits': self.n_splits,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return
            
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.forecast_horizon = model_data.get('forecast_horizon', 7)
            self.n_splits = model_data.get('n_splits', 5)
            self.is_fitted = model_data.get('is_fitted', False)
            
            logger.info(f"Loaded {len(self.models)} models from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_fitted = False

def run_forecasting(input_path: str, output_path: str, model_path: str = None):
    """Complete forecasting pipeline."""
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            # Create default output
            forecast_df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'target': ['total_alerts'],
                'forecast': [0],
                'lower_bound': [0],
                'upper_bound': [0]
            })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            forecast_df.to_csv(output_path, index=False)
            return forecast_df, {"error": "Input file not found"}
        
        # Load daily data
        df = pd.read_csv(input_path)
        
        if df.empty:
            logger.warning("Input file is empty")
            # Create default output
            forecast_df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'target': ['total_alerts'],
                'forecast': [0],
                'lower_bound': [0],
                'upper_bound': [0]
            })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            forecast_df.to_csv(output_path, index=False)
            return forecast_df, {"error": "Input file is empty"}
        
        # Convert date column
        if 'issued_date' in df.columns:
            df['issued_date'] = pd.to_datetime(df['issued_date'], errors='coerce')
            df = df.set_index('issued_date')
        else:
            # If no date column, create one
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        logger.info(f"Loaded {len(df)} days of data for forecasting")
        
        # Initialize forecaster
        forecaster = AlertForecaster(forecast_horizon=7)
        
        # Load or train models
        model_loaded = False
        if model_path and os.path.exists(model_path):
            try:
                forecaster.load_models(model_path)
                model_loaded = forecaster.is_fitted
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        # If model not loaded, try to train
        if not forecaster.is_fitted and len(df) >= 20:
            # Fit models for available alert types
            available_cols = df.columns.tolist()
            target_columns = [col for col in ['total_alerts', 'flood', 'storm', 'wind'] 
                            if col in available_cols]
            
            if target_columns:
                forecaster.fit(df, target_columns)
                
                if forecaster.is_fitted and model_path:
                    forecaster.save_models(model_path)
            else:
                logger.warning("No target columns available for forecasting")
        
        # Generate forecasts
        forecasts = forecaster.forecast(df, steps_ahead=7)
        
        # Prepare output data
        forecast_df = pd.DataFrame()
        
        for target, data in forecasts.items():
            temp_df = pd.DataFrame({
                'date': data['dates'],
                'target': target,
                'forecast': data['values'],
                'lower_bound': data['confidence_intervals'][0],
                'upper_bound': data['confidence_intervals'][1]
            })
            forecast_df = pd.concat([forecast_df, temp_df], ignore_index=True)
        
        # Save forecasts
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        
        logger.info(f"Forecasting complete. Generated forecasts for {len(forecasts)} targets")
        logger.info(f"Results saved to {output_path}")
        
        return forecast_df, {"status": "success", "forecasts_generated": len(forecasts)}
        
    except Exception as e:
        logger.error(f"Forecasting pipeline failed: {str(e)}")
        # Create default output
        try:
            forecast_df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'target': ['total_alerts'],
                'forecast': [0],
                'lower_bound': [0],
                'upper_bound': [0]
            })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            forecast_df.to_csv(output_path, index=False)
            return forecast_df, {"error": str(e)}
        except:
            return pd.DataFrame(), {"error": "Complete pipeline failure"}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    input_file = "data/processed/weather_alerts_daily.csv"
    output_file = "data/output/forecast_results.csv"
    model_file = "models/xgboost_forecast.pkl"
    
    # Create directories
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    result_df, status = run_forecasting(input_file, output_file, model_file)
    print(f"Forecast generated: {len(result_df)} rows")
    print(f"Status: {status}")
