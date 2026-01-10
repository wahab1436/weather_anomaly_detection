#!/usr/bin/env python3
"""
Weather Anomaly Detection Dashboard - Streamlit App
Separated from main system logic for better organization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import time
import importlib
import plotly.graph_objects as go

# Add src to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Set page config
st.set_page_config(
    page_title="Weather Anomaly Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #3B82F6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --background-primary: #FFFFFF;
        --background-secondary: #F9FAFB;
        --background-tertiary: #F3F4F6;
        --border-color: #E5E7EB;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--secondary-color);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid var(--secondary-color);
    }
    
    .metric-card {
        background-color: var(--background-secondary);
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data():
    """Load data from backend processing."""
    data = {
        'daily_stats': pd.DataFrame(),
        'anomalies': pd.DataFrame(),
        'forecasts': pd.DataFrame(),
        'alerts': pd.DataFrame(),
        'insights': [],
        'system_status': {}
    }
    
    # Try to load from data files
    data_files = {
        'daily_stats': 'data/processed/weather_alerts_daily.csv',
        'anomalies': 'data/output/anomaly_results.csv',
        'forecasts': 'data/output/forecast_results.csv',
        'alerts': 'data/processed/weather_alerts_processed.csv'
    }
    
    for key, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    # Convert date columns
                    date_cols = ['issued_date', 'date', 'timestamp']
                    for col in date_cols:
                        if col in df.columns:
                            df['date'] = pd.to_datetime(df[col], errors='coerce')
                            df.set_index('date', inplace=True, drop=False)
                            break
                    data[key] = df
            except Exception as e:
                st.warning(f"Error loading {filepath}: {str(e)[:100]}")
    
    # Load insights
    insight_files = [
        'data/output/anomaly_results_explanations.json',
        'data/output/insights.json'
    ]
    
    for filepath in insight_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    insights_data = json.load(f)
                    if isinstance(insights_data, dict):
                        if 'insights' in insights_data:
                            data['insights'] = insights_data['insights']
                        elif 'message' in insights_data:
                            data['insights'] = [insights_data['message']]
            except:
                pass
    
    # Create demo data if no real data found
    if data['daily_stats'].empty:
        data = create_demo_data()
        data['system_status'] = {'data_source': 'Demo Data', 'data_quality': 'low'}
    else:
        data['system_status'] = {'data_source': 'Live Data', 'data_quality': 'high'}
    
    return data

def create_demo_data():
    """Create demo data for display."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Daily stats
    total_alerts = np.clip(np.random.normal(30, 8, len(dates)), 10, 60).astype(int)
    
    daily_stats = pd.DataFrame({
        'date': dates,
        'total_alerts': total_alerts,
        'flood': np.random.poisson(3, len(dates)),
        'storm': np.random.poisson(5, len(dates)),
        'wind': np.random.poisson(4, len(dates)),
        'severity_score': np.clip(np.random.normal(0.6, 0.15, len(dates)), 0.1, 1.0),
        '7_day_avg': pd.Series(total_alerts).rolling(7, min_periods=1).mean().values,
        'day_over_day_change': pd.Series(total_alerts).pct_change().fillna(0).values * 100
    })
    daily_stats.set_index('date', inplace=True)
    
    # Anomalies
    anomalies = daily_stats.copy()
    anomalies['is_anomaly'] = False
    anomalies['anomaly_score'] = np.random.uniform(0, 0.3, len(dates))
    anomalies['anomaly_severity'] = 'low'
    
    # Add a few anomalies
    anomaly_indices = np.random.choice(range(15, 30), 3, replace=False)
    for idx in anomaly_indices:
        anomalies.iloc[idx, anomalies.columns.get_loc('is_anomaly')] = True
        anomalies.iloc[idx, anomalies.columns.get_loc('anomaly_score')] = np.random.uniform(0.7, 0.9)
        anomalies.iloc[idx, anomalies.columns.get_loc('anomaly_severity')] = 'medium'
    
    # Forecasts
    forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7, freq='D')
    forecasts = pd.DataFrame({
        'date': forecast_dates,
        'target': 'total_alerts',
        'forecast': np.clip(total_alerts[-1] + np.cumsum(np.random.normal(0, 1.5, 7)), 10, 50).astype(int),
        'lower_bound': np.clip(total_alerts[-1] + np.cumsum(np.random.normal(-2, 1, 7)), 5, 45).astype(int),
        'upper_bound': np.clip(total_alerts[-1] + np.cumsum(np.random.normal(2, 1, 7)), 15, 55).astype(int)
    })
    
    # Alerts
    alerts = pd.DataFrame([{
        'alert_id': f'DEMO_{i}',
        'headline': f'Weather Alert {i}',
        'severity': np.random.choice(['Minor', 'Moderate', 'Severe']),
        'alert_type': np.random.choice(['flood', 'storm', 'wind', 'other']),
        'issued_date': np.random.choice(dates).strftime('%Y-%m-%d')
    } for i in range(100)])
    
    insights = [
        f"Detected {len(anomaly_indices)} anomalies in the past 30 days",
        "Weather patterns are within normal seasonal ranges",
        "System is monitoring for unusual weather events"
    ]
    
    return {
        'daily_stats': daily_stats,
        'anomalies': anomalies,
        'forecasts': forecasts,
        'alerts': alerts,
        'insights': insights
    }

def create_timeline_chart(daily_stats, anomalies):
    """Create alert timeline chart."""
    fig = go.Figure()
    
    if not daily_stats.empty and 'total_alerts' in daily_stats.columns:
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['total_alerts'],
            mode='lines',
            name='Total Alerts',
            line=dict(color='#3B82F6', width=3)
        ))
    
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly'] == True]
        if not anomaly_points.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points['total_alerts'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#EF4444', size=10, symbol='diamond')
            ))
    
    fig.update_layout(
        title='Daily Weather Alert Trends',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_forecast_chart(forecasts):
    """Create forecast chart."""
    fig = go.Figure()
    
    if forecasts.empty:
        return fig
    
    total_forecast = forecasts[forecasts['target'] == 'total_alerts']
    if total_forecast.empty:
        total_forecast = forecasts
    
    fig.add_trace(go.Scatter(
        x=total_forecast['date'],
        y=total_forecast['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#3B82F6', width=3)
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
    st.markdown('<h1 class="main-header">Weather Anomaly Detection System</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    system_status = data.get('system_status', {})
    
    # Sidebar
    with st.sidebar:
        st.markdown('### System Controls')
        
        if st.button('Refresh Data', use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown('### System Status')
        st.info(f"Data Source: {system_status.get('data_source', 'Demo Data')}")
        
        if not data['daily_stats'].empty:
            total_days = len(data['daily_stats'])
            st.metric("Days of Data", total_days)
            
            if 'total_alerts' in data['daily_stats'].columns:
                avg_alerts = data['daily_stats']['total_alerts'].mean()
                st.metric("Avg Daily Alerts", f"{avg_alerts:.1f}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(['Dashboard', 'Anomalies', 'Forecasts'])
    
    with tab1:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
                total_alerts = data['daily_stats']['total_alerts'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Alerts</h3>
                    <div class="value">{int(total_alerts)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
                anomaly_count = data['anomalies']['is_anomaly'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Detected Anomalies</h3>
                    <div class="value">{int(anomaly_count)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if not data['forecasts'].empty:
                next_forecast = data['forecasts'].iloc[0]['forecast'] if len(data['forecasts']) > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Tomorrow's Forecast</h3>
                    <div class="value">{int(next_forecast)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts
        st.markdown('<h2 class="section-header">Trend Analysis</h2>', unsafe_allow_html=True)
        
        timeline_chart = create_timeline_chart(data['daily_stats'], data['anomalies'])
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Alert Distribution</h3>', unsafe_allow_html=True)
            if not data['daily_stats'].empty:
                alert_cols = [col for col in ['flood', 'storm', 'wind'] if col in data['daily_stats'].columns]
                if alert_cols:
                    recent_totals = data['daily_stats'][alert_cols].tail(7).sum()
                    st.bar_chart(recent_totals)
        
        with col2:
            st.markdown('<h3 class="section-header">7-Day Forecast</h3>', unsafe_allow_html=True)
            forecast_chart = create_forecast_chart(data['forecasts'])
            st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Insights
        if data['insights']:
            st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
            for insight in data['insights']:
                st.info(insight)
    
    with tab2:
        st.markdown('<h2 class="section-header">Anomaly Analysis</h2>', unsafe_allow_html=True)
        
        if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomalies = data['anomalies'][data['anomalies']['is_anomaly'] == True]
            
            if not anomalies.empty:
                st.write(f"Found {len(anomalies)} anomalies:")
                
                for idx, (date, row) in enumerate(anomalies.iterrows()):
                    with st.expander(f"Anomaly {idx+1} - {date.strftime('%Y-%m-%d')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Alerts", row.get('total_alerts', 'N/A'))
                            st.metric("Anomaly Score", f"{row.get('anomaly_score', 0):.3f}")
                        with col2:
                            st.metric("Severity", row.get('anomaly_severity', 'low').title())
                            st.metric("Confidence", f"{row.get('anomaly_confidence', 0):.3f}")
            else:
                st.success("No anomalies detected in the current data.")
            
            # Anomaly data table
            st.markdown('<h3 class="section-header">Anomaly Data</h3>', unsafe_allow_html=True)
            display_cols = [col for col in ['total_alerts', 'is_anomaly', 'anomaly_score', 'anomaly_severity'] 
                          if col in data['anomalies'].columns]
            if display_cols:
                st.dataframe(data['anomalies'][display_cols].tail(30), use_container_width=True)
        else:
            st.warning("Anomaly data not available. Run anomaly detection pipeline first.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Weather Forecasts</h2>', unsafe_allow_html=True)
        
        if not data['forecasts'].empty:
            st.dataframe(data['forecasts'], use_container_width=True)
            
            # Forecast summary
            col1, col2 = st.columns(2)
            with col1:
                if 'forecast' in data['forecasts'].columns:
                    avg_forecast = data['forecasts']['forecast'].mean()
                    st.metric("Average Forecast", f"{avg_forecast:.1f}")
            with col2:
                if 'forecast' in data['forecasts'].columns:
                    max_forecast = data['forecasts']['forecast'].max()
                    st.metric("Maximum Forecast", f"{max_forecast:.1f}")
        else:
            st.warning("Forecast data not available. Run forecasting pipeline first.")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Weather Anomaly Detection System")

if __name__ == "__main__":
    main()
