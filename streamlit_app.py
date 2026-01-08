"""
Streamlit App for Weather Anomaly Detection Dashboard
Minimal version for Streamlit Cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from pathlib import Path

# Create directories
for dir_path in ['data/raw', 'data/processed', 'data/output', 'models', 'logs']:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Simple configuration
class Config:
    PROCESSED_DATA_PATH = "data/processed/weather_alerts_processed.csv"
    ANOMALY_OUTPUT_PATH = "data/output/anomaly_results.csv"
    FORECAST_OUTPUT_PATH = "data/output/forecast_results.csv"
    DASHBOARD_PORT = 8501

# Set page config
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    layout="wide"
)

# Dashboard title
st.title("Weather Anomaly Detection Dashboard")
st.markdown("Real-time monitoring and forecasting of weather alert anomalies")

# Check if data exists
data_exists = (
    os.path.exists(Config.PROCESSED_DATA_PATH) and
    os.path.exists(Config.ANOMALY_OUTPUT_PATH) and
    os.path.exists(Config.FORECAST_OUTPUT_PATH)
)

if not data_exists:
    st.warning("No data found. Running in demo mode with sample data.")
    
    # Generate sample data for demo
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sample_alerts = pd.DataFrame({
        'timestamp': dates,
        'title': ['Weather Alert'] * len(dates),
        'description': ['Sample alert description'] * len(dates),
        'region': ['Northeast', 'Southeast', 'Midwest', 'West'] * (len(dates)//4),
        'alert_type': ['storm', 'flood', 'fire', 'wind'] * (len(dates)//4),
        'severity': ['warning', 'watch', 'advisory'] * (len(dates)//3)
    })
    
    sample_anomalies = pd.DataFrame({
        'date': dates,
        'total_alerts': np.random.randint(5, 50, size=len(dates)),
        'is_anomaly': np.random.choice([True, False], size=len(dates), p=[0.1, 0.9]),
        'anomaly_score': np.random.randn(len(dates))
    })
    sample_anomalies.set_index('date', inplace=True)
    
    sample_forecast = pd.DataFrame({
        'date': pd.date_range(start='2024-02-01', end='2024-02-07', freq='D'),
        'forecast': np.random.randint(10, 40, size=7),
        'lower_bound': np.random.randint(5, 30, size=7),
        'upper_bound': np.random.randint(15, 50, size=7)
    })
    
    df_alerts = sample_alerts
    df_anomalies = sample_anomalies
    df_forecast = sample_forecast
    
    st.info("Displaying sample data. Run the pipeline for real weather alerts.")
else:
    # Load real data
    try:
        df_alerts = pd.read_csv(Config.PROCESSED_DATA_PATH)
        df_anomalies = pd.read_csv(Config.ANOMALY_OUTPUT_PATH, index_col=0)
        df_forecast = pd.read_csv(Config.FORECAST_OUTPUT_PATH)
        
        # Convert dates
        if 'timestamp' in df_alerts.columns:
            df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
        if df_anomalies.index.name == 'date':
            df_anomalies.index = pd.to_datetime(df_anomalies.index)
        if 'date' in df_forecast.columns:
            df_forecast['date'] = pd.to_datetime(df_forecast['date'])
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Display metrics
st.subheader("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Alerts", len(df_alerts))

with col2:
    if 'timestamp' in df_alerts.columns:
        recent = df_alerts[df_alerts['timestamp'] > (datetime.now() - timedelta(days=7))]
        st.metric("7-Day Alerts", len(recent))

with col3:
    if 'is_anomaly' in df_anomalies.columns:
        st.metric("Anomalies", df_anomalies['is_anomaly'].sum())

with col4:
    if 'forecast' in df_forecast.columns:
        st.metric("Forecast Avg", f"{df_forecast['forecast'].mean():.1f}")

st.markdown("---")

# Alert trend chart
st.subheader("Alert Trends")
if 'total_alerts' in df_anomalies.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_anomalies.index,
        y=df_anomalies['total_alerts'],
        mode='lines',
        name='Daily Alerts'
    ))
    
    # Add anomalies if present
    if 'is_anomaly' in df_anomalies.columns:
        anomalies = df_anomalies[df_anomalies['is_anomaly']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['total_alerts'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
    
    # Add forecast if available
    if not df_forecast.empty and 'date' in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast['date'],
            y=df_forecast['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(dash='dash')
        ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Alert type distribution
if 'alert_type' in df_alerts.columns:
    st.subheader("Alert Type Distribution")
    alert_counts = df_alerts['alert_type'].value_counts()
    fig = px.bar(
        x=alert_counts.index,
        y=alert_counts.values,
        labels={'x': 'Alert Type', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Data Source**: National Weather Service (weather.gov)  
**Update Frequency**: Hourly  
**Dashboard Version**: 1.0.0
""")
