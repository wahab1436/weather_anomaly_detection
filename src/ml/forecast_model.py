"""
Forecasting module for weather alert predictions.
Uses XGBoost for time series forecasting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

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
        
        # Target variable
        features_df['target'] = features_df[target_col]
        
        # Lag features
        for lag in range(1, 31):  # Last 30 days
            features_df[f'lag_{lag}'] = features_df['target'].shift(lag)
        
        # Rolling statistics
        windows = [3, 7, 14, 30]
        for window in windows:
            features_df[f'rolling_mean_{window}'] = features_df['target'].rolling(window=window).mean()
            features_df[f'rolling_std_{window}'] = features_df['target'].rolling(window=window).std()
            features_df[f'rolling_min_{window}'] = features_df['target'].rolling(window=window).min()
            features_df[f'rolling_max_{window}'] = features_df['target'].rolling(window=window).max()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            features_df[f'ema_{alpha}'] = features_df['target'].ewm(alpha=alpha).mean()
        
        # Temporal features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_month'] = features_df.index.day
        features_df['week_of_year'] = features_df.index.isocalendar().week
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['year'] = features_df.index.year
        
        # Cyclical encoding for temporal features
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Difference features
        features_df['diff_1'] = features_df['target'].diff(1)
        features_df['diff_7'] = features_df['target'].diff(7)
        
        # Percentage changes
        features_df['pct_change_1'] = features_df['target'].pct_change(1)
        features_df['pct_change_7'] = features_df['target'].pct_change(7)
        
        # Remove rows with NaN values (from lag features)
        features_df = features_df.dropna()
        
        # Define feature columns (all except target)
        feature_columns = [col for col in features_df.columns if col != 'target']
        
        return features_df, feature_columns
    
    def prepare_forecast_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
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
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, df: pd.DataFrame, target_columns: List[str] = None):
        """Fit forecasting models for specified target columns."""
        if target_columns is None:
            target_columns = ['total_alerts', 'flood', 'storm', 'wind']
        
        for target_col in target_columns:
            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found in dataframe")
                continue
            
            try:
                # Prepare data
                X, y, feature_cols = self.prepare_forecast_data(df, target_col)
                
                if len(X) < 100:
                    logger.warning(f"Insufficient data for {target_col}. Need at least 100 samples.")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = self.train_test_split_time_series(X, y, test_size=0.2)
                
                # Initialize and train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=50
                )
                
                # Train with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
                
                # Store model
                self.models[target_col] = {
                    'model': model,
                    'feature_columns': feature_cols,
                    'last_date': df.index[-1] if hasattr(df, 'index') else pd.Timestamp.now()
                }
                
                # Evaluate model
                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
                logger.info(f"Model for {target_col}: MAE={mae:.2f}, RMSE={rmse:.2f}")
                
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
                last_date = latest_data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps_ahead,
                    freq='D'
                )
                
                # Prepare initial feature vector
                current_features = self._get_latest_features(latest_data, target_col, feature_cols)
                
                if current_features is None:
                    continue
                
                # Generate forecasts recursively
                target_forecasts = []
                feature_history = current_features.copy()
                
                for i in range(steps_ahead):
                    # Predict next value
                    next_pred = model.predict(feature_history.reshape(1, -1))[0]
                    target_forecasts.append(max(0, next_pred))  # Ensure non-negative
                    
                    # Update features for next prediction
                    feature_history = self._update_features(
                        feature_history, next_pred, feature_cols, i+1
                    )
                
                # Store forecasts
                forecasts[target_col] = {
                    'dates': forecast_dates,
                    'values': target_forecasts,
                    'confidence_intervals': self._calculate_confidence_intervals(target_forecasts)
                }
                
            except Exception as e:
                logger.error(f"Error forecasting {target_col}: {str(e)}")
        
        return forecasts
    
    def _get_latest_features(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> np.ndarray:
        """Get the latest feature vector for forecasting."""
        try:
            # Create features for the latest data point
            features_df, _ = self.create_features(df, target_col)
            
            if features_df.empty or feature_cols[0] not in features_df.columns:
                return None
            
            # Get the latest feature vector
            latest_features = features_df[feature_cols].iloc[-1].values
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None
    
    def _update_features(self, features: np.ndarray, new_value: float, 
                        feature_cols: List[str], step: int) -> np.ndarray:
        """Update feature vector with new prediction for recursive forecasting."""
        # This is a simplified version - in practice, you'd need to update
        # all lag features, rolling statistics, etc.
        updated_features = features.copy()
        
        # Find indices of lag features
        lag_indices = [i for i, col in enumerate(feature_cols) if col.startswith('lag_')]
        
        # Shift lag features
        for idx in sorted(lag_indices, reverse=True):
            lag_num = int(feature_cols[idx].split('_')[1])
            if lag_num == 1:
                updated_features[idx] = new_value
            elif lag_num - 1 in [int(feature_cols[i].split('_')[1]) for i in lag_indices]:
                # Find the index for lag_{lag_num-1}
                prev_idx = next(i for i, col in enumerate(feature_cols) 
                              if col == f'lag_{lag_num-1}')
                updated_features[idx] = updated_features[prev_idx]
        
        return updated_features
    
    def _calculate_confidence_intervals(self, predictions: List[float], 
                                      confidence_level: float = 0.95) -> Tuple[List, List]:
        """Calculate confidence intervals for predictions."""
        # Simplified confidence interval calculation
        # In production, use proper uncertainty quantification methods
        std = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
        z_score = 1.96  # For 95% confidence
        
        lower = [max(0, p - z_score * std) for p in predictions]
        upper = [p + z_score * std for p in predictions]
        
        return lower, upper
    
    def evaluate_models(self, df: pd.DataFrame) -> Dict:
        """Evaluate model performance using time series cross-validation."""
        evaluation_results = {}
        
        for target_col, model_data in self.models.items():
            try:
                # Prepare data
                X, y, _ = self.prepare_forecast_data(df, target_col)
                
                if len(X) < 100:
                    continue
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                
                metrics = {
                    'mae': [],
                    'rmse': [],
                    'mape': []
                }
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train model on fold
                    fold_model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.01,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    fold_model.fit(X_train, y_train, verbose=False)
                    
                    # Predict and evaluate
                    y_pred = fold_model.predict(X_test)
                    
                    # Calculate metrics
                    metrics['mae'].append(mean_absolute_error(y_test, y_pred))
                    metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    
                    # Calculate MAPE (handle zero values)
                    mask = y_test != 0
                    if mask.any():
                        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                        metrics['mape'].append(mape)
                
                # Aggregate metrics
                evaluation_results[target_col] = {
                    'mean_mae': np.mean(metrics['mae']),
                    'std_mae': np.std(metrics['mae']),
                    'mean_rmse': np.mean(metrics['rmse']),
                    'std_rmse': np.std(metrics['rmse']),
                    'mean_mape': np.mean(metrics['mape']) if metrics['mape'] else None,
                    'n_folds': self.n_splits
                }
                
            except Exception as e:
                logger.error(f"Error evaluating model for {target_col}: {str(e)}")
        
        return evaluation_results
    
    def save_models(self, filepath: str):
        """Save all trained models."""
        if not self.is_fitted:
            logger.warning("No models fitted. Nothing to save.")
            return
        
        model_data = {
            'models': self.models,
            'forecast_horizon': self.forecast_horizon,
            'n_splits': self.n_splits,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.forecast_horizon = model_data['forecast_horizon']
            self.n_splits = model_data['n_splits']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Loaded {len(self.models)} models from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_fitted = False

def run_forecasting(input_path: str, output_path: str, model_path: str = None):
    """Complete forecasting pipeline."""
    try:
        # Load daily data
        df = pd.read_csv(input_path)
        df['issued_date'] = pd.to_datetime(df['issued_date'])
        df = df.set_index('issued_date')
        
        logger.info(f"Loaded {len(df)} days of data for forecasting")
        
        # Initialize forecaster
        forecaster = AlertForecaster(forecast_horizon=7)
        
        # Load or train models
        if model_path and os.path.exists(model_path):
            forecaster.load_models(model_path)
        else:
            # Fit models for key alert types
            target_columns = ['total_alerts', 'flood', 'storm', 'wind']
            forecaster.fit(df, target_columns)
            
            if model_path:
                forecaster.save_models(model_path)
        
        # Generate forecasts
        forecasts = forecaster.forecast(df, steps_ahead=7)
        
        # Evaluate models
        evaluation_results = forecaster.evaluate_models(df)
        
        # Prepare output data
        output_data = {
            'forecasts': forecasts,
            'evaluation': evaluation_results,
            'last_training_date': df.index[-1].strftime('%Y-%m-%d'),
            'forecast_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save forecasts
        forecast_df = pd.DataFrame()
        
        for target, data in forecasts.items():
            temp_df = pd.DataFrame({
                'date': data['dates'],
                'target': target,
                'forecast': data['values'],
                'lower_bound': data['confidence_intervals'][0],
                'upper_bound': data['confidence_intervals'][1]
            })
            forecast_df = pd.concat([forecast_df, temp_df])
        
        forecast_df.to_csv(output_path, index=False)
        
        # Save evaluation results
        eval_path = output_path.replace('.csv', '_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Forecasting complete. Generated forecasts for {len(forecasts)} targets")
        logger.info(f"Results saved to {output_path}")
        
        return forecast_df, evaluation_results
        
    except Exception as e:
        logger.error(f"Forecasting pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/weather_alerts_daily.csv"
    output_file = "data/output/forecast_results.csv"
    model_file = "models/xgboost_forecast.pkl"
    
    run_forecasting(input_file, output_file, model_file)
