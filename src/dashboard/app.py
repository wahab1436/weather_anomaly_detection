"""
Streamlit Dashboard for Weather Anomaly Detection
Standalone app without CLI dependency.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
import logging

# ---------------------- Configuration ----------------------
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Directories ----------------------
def setup_directories():
    dirs = ['data/raw', 'data/processed', 'data/output', 'models', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
setup_directories()

# ---------------------- Data Loading ----------------------
@st.cache_data(ttl=3600)
def load_data():
    files = {
        'daily_stats': 'data/processed/weather_alerts_daily.csv',
        'anomalies': 'data/output/anomaly_results.csv',
        'forecasts': 'data/output/forecast_results.csv',
        'alerts': 'data/processed/weather_alerts_processed.csv',
        'insights': 'data/output/insights.json'
    }
    data = {}
    for key, path in files.items():
        try:
            if os.path.exists(path):
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    elif 'issued_date' in df.columns:
                        df['date'] = pd.to_datetime(df['issued_date'])
                        df = df.set_index('date')
                    data[key] = df
                else:
                    with open(path, 'r') as f:
                        data[key] = json.load(f)
            else:
                data[key] = pd.DataFrame() if path.endswith(".csv") else {}
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            data[key] = pd.DataFrame() if path.endswith(".csv") else {}
    return data

# ---------------------- Dashboard Components ----------------------
def create_alert_timeline(daily_stats, anomalies):
    if daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['total_alerts'],
                             name='Total Alerts', line=dict(color='#3B82F6', width=2)))

    if '7_day_avg' in daily_stats.columns:
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['7_day_avg'],
                                 name='7-Day Avg', line=dict(color='#6B7280', width=1, dash='dash')))

    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        points = anomalies[anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(x=points.index, y=points['total_alerts'], name='Anomalies',
                                 mode='markers', marker=dict(color='#DC2626', size=10, symbol='diamond')))

    if 'severity_score' in daily_stats.columns:
        fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['severity_score'] * 100,
                                 name='Severity (%)', line=dict(color='#EF4444', width=1), mode='lines'),
                      secondary_y=True)

    fig.update_layout(title='Daily Alerts with Anomaly Detection',
                      xaxis_title='Date', yaxis_title='Number of Alerts',
                      yaxis2_title='Severity (%)', template='plotly_white', height=500,
                      hovermode='x unified')
    return fig

def create_forecast_chart(forecasts):
    if forecasts.empty:
        fig = go.Figure()
        fig.add_annotation(text="No forecast data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    if 'date' in forecasts.columns:
        forecasts['date'] = pd.to_datetime(forecasts['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecasts['date'], y=forecasts['forecast'],
                             name='Forecast', line=dict(color='#3B82F6', width=3)))
    if 'lower_bound' in forecasts.columns and 'upper_bound' in forecasts.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecasts['date'], forecasts['date'][::-1]]),
            y=pd.concat([forecasts['upper_bound'], forecasts['lower_bound'][::-1]]),
            fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Confidence'))
    fig.update_layout(title='7-Day Alert Forecast', xaxis_title='Date', yaxis_title='Predicted Alerts',
                      template='plotly_white', height=400, hovermode='x unified')
    return fig

def display_metrics(daily_stats):
    if daily_stats.empty:
        st.warning("No metrics to display")
        return

    total_alerts = daily_stats['total_alerts'].sum() if 'total_alerts' in daily_stats.columns else 0
    avg_daily = daily_stats['total_alerts'].mean() if 'total_alerts' in daily_stats.columns else 0
    max_daily = daily_stats['total_alerts'].max() if 'total_alerts' in daily_stats.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", f"{int(total_alerts):,}")
    col2.metric("Avg Daily Alerts", f"{avg_daily:.1f}")
    col3.metric("Max Daily Alerts", f"{max_daily}")

# ---------------------- Main Dashboard ----------------------
def main():
    data = load_data()

    st.title("Weather Anomaly Detection Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()

    # Overview Tab
    tab1, tab2 = st.tabs(["Overview", "Forecasts"])

    with tab1:
        st.header("Overview")
        display_metrics(data['daily_stats'])
        fig_alerts = create_alert_timeline(data['daily_stats'], data['anomalies'])
        st.plotly_chart(fig_alerts, use_container_width=True)

    with tab2:
        st.header("Forecasts")
        fig_forecast = create_forecast_chart(data['forecasts'])
        st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("---")
    st.caption(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
