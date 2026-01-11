"""
Weather Anomaly Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root / 'src'))

# Set page config
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data():
    """Load all data from the pipeline."""
    data = {
        'daily_stats': pd.DataFrame(),
        'anomalies': pd.DataFrame(),
        'forecasts': pd.DataFrame(),
        'alerts': pd.DataFrame()
    }
    
    # Load daily stats
    daily_path = "data/processed/weather_alerts_daily.csv"
    if os.path.exists(daily_path):
        df = pd.read_csv(daily_path)
        if not df.empty:
            if 'issued_date' in df.columns:
                df['issued_date'] = pd.to_datetime(df['issued_date'], errors='coerce')
                df.set_index('issued_date', inplace=True)
            data['daily_stats'] = df
    
    # Load anomalies
    anomaly_path = "data/output/anomaly_results.csv"
    if os.path.exists(anomaly_path):
        df = pd.read_csv(anomaly_path)
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.set_index('date', inplace=True)
            data['anomalies'] = df
    
    # Load forecasts
    forecast_path = "data/output/forecast_results.csv"
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        if not df.empty:
            data['forecasts'] = df
    
    # Load alerts
    alert_path = "data/processed/weather_alerts_processed.csv"
    if os.path.exists(alert_path):
        df = pd.read_csv(alert_path)
        if not df.empty:
            data['alerts'] = df
    
    return data

def create_timeline_chart(daily_stats, anomalies):
    """Create alert timeline with anomalies."""
    if daily_stats.empty:
        return None
    
    fig = go.Figure()
    
    if 'total_alerts' in daily_stats.columns:
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['total_alerts'],
            name='Total Alerts',
            line=dict(color='#3B82F6', width=3),
            mode='lines'
        ))
    
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly']]
        if not anomaly_points.empty and 'total_alerts' in anomaly_points.columns:
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points['total_alerts'],
                name='Anomalies',
                mode='markers',
                marker=dict(
                    color='#EF4444',
                    size=10,
                    symbol='diamond'
                )
            ))
    
    fig.update_layout(
        title='Weather Alert Timeline',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_forecast_chart(forecasts):
    """Create forecast visualization."""
    if forecasts.empty or 'date' not in forecasts.columns:
        return None
    
    # Filter for total alerts forecast
    if 'target' in forecasts.columns:
        total_forecast = forecasts[forecasts['target'] == 'total_alerts']
    else:
        total_forecast = forecasts.head(7)
    
    if total_forecast.empty:
        return None
    
    # Convert date column
    total_forecast['date'] = pd.to_datetime(total_forecast['date'], errors='coerce')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=total_forecast['date'],
        y=total_forecast['forecast'],
        name='Forecast',
        line=dict(color='#3B82F6', width=3),
        mode='lines+markers'
    ))
    
    if 'lower_bound' in total_forecast.columns and 'upper_bound' in total_forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([total_forecast['date'], total_forecast['date'][::-1]]),
            y=pd.concat([total_forecast['upper_bound'], total_forecast['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title='7-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Alerts',
        template='plotly_white',
        height=350
    )
    
    return fig

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">Weather Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    daily_stats = data['daily_stats']
    anomalies = data['anomalies']
    forecasts = data['forecasts']
    alerts = data['alerts']
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Dashboard Controls")
        
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Data status
        total_days = len(daily_stats) if not daily_stats.empty else 0
        total_alerts = len(alerts) if not alerts.empty else 0
        anomaly_count = anomalies['is_anomaly'].sum() if not anomalies.empty and 'is_anomaly' in anomalies.columns else 0
        
        st.info(f"Data Days: {total_days}")
        st.info(f"Total Alerts: {total_alerts}")
        st.info(f"Anomalies: {anomaly_count}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomalies", "Forecasts", "Data"])
    
    with tab1:
        # Overview tab
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not daily_stats.empty and 'total_alerts' in daily_stats.columns:
                total = daily_stats['total_alerts'].sum()
                st.metric("Total Alerts", f"{int(total):,}")
        
        with col2:
            if not daily_stats.empty and 'total_alerts' in daily_stats.columns:
                avg = daily_stats['total_alerts'].mean()
                st.metric("Avg Daily", f"{avg:.1f}")
        
        with col3:
            if not anomalies.empty and 'is_anomaly' in anomalies.columns:
                st.metric("Anomalies", int(anomaly_count))
        
        with col4:
            if not forecasts.empty:
                next_day = forecasts.iloc[0]['forecast'] if len(forecasts) > 0 else 0
                st.metric("Tomorrow", int(next_day))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            timeline_chart = create_timeline_chart(daily_stats, anomalies)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
        
        with col2:
            forecast_chart = create_forecast_chart(forecasts)
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
    
    with tab2:
        # Anomalies tab
        if not anomalies.empty and 'is_anomaly' in anomalies.columns:
            anomaly_data = anomalies[anomalies['is_anomaly']]
            
            if not anomaly_data.empty:
                st.dataframe(anomaly_data[['total_alerts', 'anomaly_score']], use_container_width=True)
            else:
                st.info("No anomalies detected")
        else:
            st.info("Run anomaly detection first")
    
    with tab3:
        # Forecasts tab
        if not forecasts.empty:
            st.dataframe(forecasts, use_container_width=True)
        else:
            st.info("Run forecasting first")
    
    with tab4:
        # Data tab
        if not alerts.empty:
            st.dataframe(alerts.head(20), use_container_width=True)
        else:
            st.info("No alert data available")
    
    # Footer
    st.markdown("---")
    st.caption(f"Weather Anomaly Detection System | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
