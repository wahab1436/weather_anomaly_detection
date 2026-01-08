"""
Forecast model for predicting future weather alerts
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import logging
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
        logging.FileHandler('logs/forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertForecaster:
    """Forecast future weather alerts"""
    
    def __init__(self, target_col: str = 'total_alerts', forecast_horizon: int = None):
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon or Config.FORECAST_HORIZON
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.lag_features = []
        self.rolling_features = []
        self.is_fitted = False
        
        # XGBoost parameters
        self.xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def create_lag_features(self, df: pd.DataFrame, max_lag: int = 14) -> pd.DataFrame:
        """Create lag features for time series"""
        df_lagged = df.copy()
        self.lag_features = []
        
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
            if lag <= len(df) // 2:  # Only create if we have enough data
                df_lagged[f'{self.target_col}_lag_{lag}'] = df_lagged[self.target_col].shift(lag)
                self.lag_features.append(f'{self.target_col}_lag_{lag}')
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics features"""
        df_rolling = df.copy()
        self.rolling_features = []
        
        windows = [3, 7, 14, 30]
        
        for window in windows:
            if window < len(df):
                # Rolling mean
                df_rolling[f'{self.target_col}_roll_mean_{window}'] = (
                    df_rolling[self.target_col].rolling(window=window, min_periods=1).mean()
                )
                self.rolling_features.append(f'{self.target_col}_roll_mean_{window}')
                
                # Rolling standard deviation
                df_rolling[f'{self.target_col}_roll_std_{window}'] = (
                    df_rolling[self.target_col].rolling(window=window, min_periods=1).std()
                )
                self.rolling_features.append(f'{self.target_col}_roll_std_{window}')
                
                # Rolling min and max
                df_rolling[f'{self.target_col}_roll_min_{window}'] = (
                    df_rolling[self.target_col].rolling(window=window, min_periods=1).min()
                )
                df_rolling[f'{self.target_col}_roll_max_{window}'] = (
                    df_rolling[self.target_col].rolling(window=window, min_periods=1).max()
                )
                self.rolling_features.extend([
                    f'{self.target_col}_roll_min_{window}',
                    f'{self.target_col}_roll_max_{window}'
                ])
        
        return df_rolling
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date index"""
        df_temp = df.copy()
        
        # Cyclical encoding for temporal features
        if hasattr(df_temp.index, 'dayofweek'):
            df_temp['day_of_week_sin'] = np.sin(2 * np.pi * df_temp.index.dayofweek / 7)
            df_temp['day_of_week_cos'] = np.cos(2 * np.pi * df_temp.index.dayofweek / 7)
        
        if hasattr(df_temp.index, 'month'):
            df_temp['month_sin'] = np.sin(2 * np.pi * df_temp.index.month / 12)
            df_temp['month_cos'] = np.cos(2 * np.pi * df_temp.index.month / 12)
        
        if hasattr(df_temp.index, 'dayofyear'):
            df_temp['day_of_year_sin'] = np.sin(2 * np.pi * df_temp.index.dayofyear / 365)
            df_temp['day_of_year_cos'] = np.cos(2 * np.pi * df_temp.index.dayofyear / 365)
        
        # Additional features
        df_temp['is_weekend'] = (df_temp.index.dayofweek >= 5).astype(int)
        df_temp['quarter'] = df_temp.index.quarter
        df_temp['is_month_start'] = (df_temp.index.day == 1).astype(int)
        df_temp['is_month_end'] = (df_temp.index.day == df_temp.index.days_in_month).astype(int)
        
        return df_temp
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for forecasting"""
        logger.info("Creating forecasting features")
        
        # Start with base features
        df_features = df.copy()
        
        # Ensure we have the target column
        if self.target_col not in df_features.columns:
            # Try to find alternative target
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.target_col = numeric_cols[0]
                logger.warning(f"Target column not found. Using {self.target_col} instead.")
            else:
                logger.error("No numeric columns found for forecasting")
                return pd.DataFrame()
        
        # Create lag features
        df_features = self.create_lag_features(df_features)
        
        # Create rolling features
        df_features = self.create_rolling_features(df_features)
        
        # Create temporal features
        df_features = self.create_temporal_features(df_features)
        
        # Drop NaN values created by lag features
        initial_len = len(df_features)
        df_features = df_features.dropna()
        dropped_count = initial_len - len(df_features)
        
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with NaN values")
        
        logger.info(f"Created {len(df_features.columns)} features total")
        return df_features
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training"""
        # Create all features
        df_features = self.create_all_features(df)
        
        if df_features.empty:
            logger.error("No features created")
            return None, None, []
        
        # Define feature columns (exclude target and non-numeric)
        exclude_cols = [self.target_col]
        self.feature_columns = [
            col for col in df_features.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col])
        ]
        
        if not self.feature_columns:
            logger.error("No feature columns selected")
            return None, None, []
        
        X = df_features[self.feature_columns].values
        y = df_features[self.target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared data: X shape {X_scaled.shape}, y shape {y.shape}")
        return X_scaled, y, self.feature_columns
    
    def train(self, df: pd.DataFrame, test_size: float = None) -> Dict:
        """Train the forecasting model"""
        if test_size is None:
            test_size = Config.TEST_SIZE
        
        logger.info(f"Training forecasting model (test size: {test_size})")
        
        # Prepare data
        X, y, feature_cols = self.prepare_training_data(df)
        
        if X is None or y is None:
            logger.error("Failed to prepare training data")
            return {}
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        
        if split_idx < 10:  # Need minimum data
            split_idx = max(10, len(X) - 5)
            logger.warning(f"Adjusting split to {split_idx} for minimum training data")
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        if len(X_train) < 10:
            logger.error("Insufficient training data")
            return {}
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        logger.info(f"Model trained. Test MAE: {metrics['test_mae']:.2f}, Test RMSE: {metrics['test_rmse']:.2f}")
        
        self.is_fitted = True
        return metrics
    
    def forecast_future(self, df: pd.DataFrame, periods: int = None) -> pd.DataFrame:
        """Generate forecasts for future periods"""
        if periods is None:
            periods = self.forecast_horizon
        
        if not self.is_fitted or self.model is None:
            logger.error("Model not trained")
            return pd.DataFrame()
        
        logger.info(f"Generating forecasts for {periods} periods")
        
        # Create a copy for forecasting
        df_forecast_base = df.copy()
        
        # Get last date
        if hasattr(df_forecast_base.index, 'max'):
            last_date = df_forecast_base.index.max()
        elif 'date' in df_forecast_base.columns:
            last_date = pd.to_datetime(df_forecast_base['date']).max()
        else:
            last_date = datetime.now()
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecasts = []
        confidence_intervals = []
        
        # Prepare initial feature vector
        df_features = self.create_all_features(df_forecast_base)
        
        if df_features.empty:
            logger.error("Could not create features for forecasting")
            return pd.DataFrame()
        
        # Get last row for initial prediction
        last_features = df_features.iloc[[-1]].copy()
        
        for i in range(periods):
            # Prepare features for prediction
            X_pred = last_features[self.feature_columns].values
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make prediction
            pred = self.model.predict(X_pred_scaled)[0]
            
            # Simple confidence interval (using test RMSE if available)
            # You could make this more sophisticated
            ci = pred * 0.2  # 20% margin
            
            forecasts.append(pred)
            confidence_intervals.append(ci)
            
            # Update features for next prediction (simulate new day)
            if i < periods - 1:
                # Shift lag features
                for lag_feature in self.lag_features:
                    if lag_feature in last_features.columns:
                        # Extract lag number
                        match = re.search(r'lag_(\d+)', lag_feature)
                        if match:
                            lag_num = int(match.group(1))
                            if lag_num > 1:
                                # Shift previous lag values
                                prev_lag = f'{self.target_col}_lag_{lag_num-1}'
                                if prev_lag in last_features.columns:
                                    last_features[lag_feature] = last_features[prev_lag]
                            elif lag_num == 1:
                                last_features[lag_feature] = pred
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': np.array(forecasts) - np.array(confidence_intervals),
            'upper_bound': np.array(forecasts) + np.array(confidence_intervals),
            'confidence_interval': confidence_intervals
        })
        
        # Ensure non-negative bounds
        forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
        forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(lower=0)
        
        logger.info(f"Generated forecasts: {forecasts}")
        return forecast_df
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if filepath is None:
            filepath = Config.FORECAST_MODEL_PATH
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_col': self.target_col,
            'lag_features': self.lag_features,
            'rolling_features': self.rolling_features,
            'is_fitted': self.is_fitted,
            'forecast_horizon': self.forecast_horizon
        }, filepath)
        logger.info(f"Saved forecasting model to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = Config.FORECAST_MODEL_PATH
        
        if os.path.exists(filepath):
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_columns = saved_data['feature_columns']
            self.target_col = saved_data['target_col']
            self.lag_features = saved_data.get('lag_features', [])
            self.rolling_features = saved_data.get('rolling_features', [])
            self.is_fitted = saved_data['is_fitted']
            self.forecast_horizon = saved_data.get('forecast_horizon', Config.FORECAST_HORIZON)
            logger.info(f"Loaded forecasting model from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")

def run_forecast_pipeline(aggregated_data_path: str = None,
                         output_path: str = None,
                         model_path: str = None):
    """Run complete forecasting pipeline"""
    logger.info("=" * 60)
    logger.info("Starting forecasting pipeline")
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)
    
    if aggregated_data_path is None:
        aggregated_data_path = Config.AGGREGATED_DATA_PATH
    if output_path is None:
        output_path = Config.FORECAST_OUTPUT_PATH
    if model_path is None:
        model_path = Config.FORECAST_MODEL_PATH
    
    try:
        # Load aggregated data
        if not os.path.exists(aggregated_data_path):
            logger.error(f"Aggregated data file not found: {aggregated_data_path}")
            return None, None
        
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
            return None, None
        
        # Initialize forecaster
        forecaster = AlertForecaster(target_col='total_alerts')
        
        # Try to load existing model
        forecaster.load_model(model_path)
        
        # Train or retrain model
        metrics = forecaster.train(df)
        
        # Generate forecasts
        forecasts = forecaster.forecast_future(df, periods=Config.FORECAST_HORIZON)
        
        if forecasts.empty:
            logger.error("No forecasts generated")
            return None, metrics
        
        # Save forecasts
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecasts.to_csv(output_path, index=False)
        
        # Save model
        forecaster.save_model(model_path)
        
        logger.info(f"Saved forecasts to {output_path}")
        logger.info(f"Forecast range: {forecasts['date'].min()} to {forecasts['date'].max()}")
        
        return forecasts, metrics
        
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run forecasting pipeline
    forecasts, metrics = run_forecast_pipeline(
        Config.AGGREGATED_DATA_PATH,
        Config.FORECAST_OUTPUT_PATH,
        Config.FORECAST_MODEL_PATH
    )
    
    if forecasts is not None and metrics is not None:
        print(f"Forecasting complete. Test MAE: {metrics['test_mae']:.2f}")
        print(f"Forecasts generated: {len(forecasts)} days")
    else:
        print("Forecasting failed")
