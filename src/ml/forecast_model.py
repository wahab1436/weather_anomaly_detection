import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from datetime import datetime, timedelta
import os
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)

class AlertForecaster:
    """Forecasts future weather alert patterns."""
    
    def __init__(self, model_type: str = 'xgboost', forecast_horizon: int = 7):
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                objective='reg:squarederror'
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_forecast_data(self, df: pd.DataFrame, target_col: str = 'total_alerts',
                            lookback: int = 14) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for time series forecasting."""
        if df.empty or len(df) < lookback + self.forecast_horizon:
            logger.warning("Insufficient data for forecasting")
            return pd.DataFrame(), pd.Series()
        
        # Ensure sorted by date
        df = df.sort_values('date').reset_index(drop=True)
        
        features_list = []
        targets_list = []
        
        for i in range(lookback, len(df) - self.forecast_horizon):
            # Historical features
            hist_data = df.iloc[i-lookback:i]
            
            # Extract numerical features
            numerical_features = hist_data.select_dtypes(include=[np.number])
            
            # Create feature vector
            feature_vector = []
            
            # Summary statistics from history
            for col in ['total_alerts']:
                if col in numerical_features.columns:
                    feature_vector.extend([
                        numerical_features[col].mean(),
                        numerical_features[col].std(),
                        numerical_features[col].max(),
                        numerical_features[col].min(),
                        numerical_features[col].iloc[-1]  # Most recent value
                    ])
            
            # Date features for prediction period
            prediction_date = df.iloc[i]['date']
            feature_vector.extend([
                prediction_date.dayofweek,
                prediction_date.month,
                prediction_date.isocalendar().week,
                1 if prediction_date.dayofweek >= 5 else 0,  # Weekend
                prediction_date.day
            ])
            
            # Add lag features for specific alert types
            alert_type_cols = [col for col in df.columns if 'alert_type' in str(col)]
            for col in alert_type_cols[:5]:  # Limit to top 5 types
                if col in numerical_features.columns:
                    feature_vector.append(numerical_features[col].iloc[-1])  # Most recent count
                    feature_vector.append(numerical_features[col].mean())    # Average
            
            features_list.append(feature_vector)
            
            # Target: future value
            target = df.iloc[i + self.forecast_horizon][target_col]
            targets_list.append(target)
        
        # Create DataFrames
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets_list)
        
        # Store feature names
        self.feature_names = [f'feature_{i}' for i in range(features_df.shape[1])]
        features_df.columns = self.feature_names
        
        return features_df, targets_series
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train the forecast model."""
        if X.empty or y.empty:
            logger.warning("No data for training")
            return
        
        # Split data (time series aware)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
        
        # Final training on all data
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Model trained with CV MAE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            logger.warning("Model not trained")
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def forecast_future(self, historical_df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """Generate future forecasts."""
        if not self.is_trained or historical_df.empty:
            logger.warning("Cannot generate forecast: model not trained or no data")
            return pd.DataFrame()
        
        # Prepare data for forecasting
        lookback = 14
        if len(historical_df) < lookback:
            lookback = len(historical_df)
        
        forecasts = []
        
        # Generate forecasts for each future day
        for day_offset in range(1, days_ahead + 1):
            # Use recent history
            recent_data = historical_df.iloc[-lookback:].copy()
            
            # Prepare features for this prediction
            feature_vector = []
            
            # Historical statistics
            numerical_features = recent_data.select_dtypes(include=[np.number])
            
            for col in ['total_alerts']:
                if col in numerical_features.columns:
                    feature_vector.extend([
                        numerical_features[col].mean(),
                        numerical_features[col].std(),
                        numerical_features[col].max(),
                        numerical_features[col].min(),
                        numerical_features[col].iloc[-1]
                    ])
            
            # Date features for forecast day
            forecast_date = historical_df['date'].iloc[-1] + timedelta(days=day_offset)
            feature_vector.extend([
                forecast_date.dayofweek,
                forecast_date.month,
                forecast_date.isocalendar().week,
                1 if forecast_date.dayofweek >= 5 else 0,
                forecast_date.day
            ])
            
            # Lag features for alert types
            alert_type_cols = [col for col in historical_df.columns if 'alert_type' in str(col)]
            for col in alert_type_cols[:5]:
                if col in numerical_features.columns:
                    feature_vector.append(numerical_features[col].iloc[-1])
                    feature_vector.append(numerical_features[col].mean())
            
            # Make prediction
            X_pred = pd.DataFrame([feature_vector], columns=self.feature_names)
            prediction = self.predict(X_pred)
            
            # Add confidence interval (simple approach)
            if len(prediction) > 0:
                confidence = prediction[0] * 0.2  # 20% margin
                
                forecasts.append({
                    'date': forecast_date,
                    'predicted_alerts': max(0, round(prediction[0])),
                    'lower_bound': max(0, round(prediction[0] - confidence)),
                    'upper_bound': max(0, round(prediction[0] + confidence)),
                    'confidence': 'medium'
                })
        
        return pd.DataFrame(forecasts)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance."""
        if not self.is_trained:
            return {}
        
        predictions = self.predict(X)
        
        if len(predictions) == 0:
            return {}
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / np.maximum(y, 1))) * 100
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if self.is_trained:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'forecast_horizon': self.forecast_horizon
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data.get('model_type', 'xgboost')
            self.forecast_horizon = model_data.get('forecast_horizon', 7)
            self.is_trained = True
            logger.info(f"Loaded model from {filepath}")
    
    def generate_forecast_insights(self, forecast_df: pd.DataFrame) -> Dict:
        """Generate plain English insights from forecasts."""
        insights = {
            'summary': '',
            'key_forecasts': [],
            'recommendations': []
        }
        
        if forecast_df.empty:
            insights['summary'] = "Insufficient data for forecasting."
            return insights
        
        # Calculate overall trend
        trend = 'stable'
        if len(forecast_df) > 1:
            first = forecast_df['predicted_alerts'].iloc[0]
            last = forecast_df['predicted_alerts'].iloc[-1]
            if last > first * 1.3:
                trend = 'increasing'
            elif last < first * 0.7:
                trend = 'decreasing'
        
        # Generate summary
        avg_prediction = forecast_df['predicted_alerts'].mean()
        max_day = forecast_df.loc[forecast_df['predicted_alerts'].idxmax()]
        
        insights['summary'] = (
            f"Forecast predicts an average of {avg_prediction:.0f} alerts per day "
            f"over the next {len(forecast_df)} days, with a {trend} trend."
        )
        
        # List key forecasts
        for _, row in forecast_df.iterrows():
            forecast_info = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'predicted_alerts': int(row['predicted_alerts']),
                'range': f"{int(row['lower_bound'])}-{int(row['upper_bound'])}",
                'confidence': row['confidence']
            }
            insights['key_forecasts'].append(forecast_info)
        
        # Generate recommendations
        if trend == 'increasing':
            insights['recommendations'].append("Increasing alert trend detected. Consider preparing for potential severe weather.")
        
        if max_day['predicted_alerts'] > 15:
            insights['recommendations'].append(f"High alert volume predicted on {max_day['date'].strftime('%Y-%m-%d')}. Monitor weather conditions closely.")
        
        if avg_prediction < 5:
            insights['recommendations'].append("Low alert volume predicted. Normal monitoring procedures are sufficient.")
        
        return insights

def main():
    """Main forecasting function."""
    forecaster = AlertForecaster(model_type='xgboost', forecast_horizon=7)
    
    # Load processed data
    processed_path = 'data/processed/weather_alerts_processed.csv'
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found: {processed_path}")
        return
    
    processed_df = pd.read_csv(processed_path, parse_dates=['date'])
    
    if processed_df.empty or len(processed_df) < 30:
        logger.warning("Insufficient historical data for forecasting (need at least 30 days)")
        return
    
    # Prepare data
    X, y = forecaster.prepare_forecast_data(processed_df)
    
    if X.empty or y.empty:
        logger.warning("Could not prepare forecast data")
        return
    
    # Train model
    forecaster.train(X, y)
    
    # Evaluate model
    metrics = forecaster.evaluate(X, y)
    logger.info(f"Model metrics: {metrics}")
    
    # Generate future forecasts
    forecast_df = forecaster.forecast_future(processed_df, days_ahead=7)
    
    if not forecast_df.empty:
        # Save forecasts
        forecast_path = 'data/output/alert_forecasts.csv'
        os.makedirs('data/output', exist_ok=True)
        forecast_df.to_csv(forecast_path, index=False)
        
        # Generate insights
        insights = forecaster.generate_forecast_insights(forecast_df)
        
        # Save insights
        insights_path = 'data/output/forecast_insights.json'
        import json
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Save model
        model_path = 'models/xgboost_forecast.pkl'
        forecaster.save_model(model_path)
        
        logger.info("Forecasting completed successfully")
        
        # Print summary
        print(f"Forecasts generated for {len(forecast_df)} days")
        print(f"Average predicted alerts: {forecast_df['predicted_alerts'].mean():.1f}")

if __name__ == "__main__":
    main()
