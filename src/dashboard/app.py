#!/usr/bin/env python3
"""
Weather Anomaly Detection Dashboard - Using Real Data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import plotly.graph_objects as go

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Page configuration
st.set_page_config(
    page_title="Weather Anomaly Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)
def load_real_data():
    """Load real data from the backend processing pipeline."""
    data = {
        'daily_stats': pd.DataFrame(),
        'anomalies': pd.DataFrame(),
        'forecasts': pd.DataFrame(),
        'alerts': pd.DataFrame(),
        'insights': [],
        'system_status': {}
    }
    
    try:
        # Load daily statistics from real processed data
        daily_path = os.path.join(project_root, "data", "processed", "weather_alerts_daily.csv")
        if os.path.exists(daily_path):
            df = pd.read_csv(daily_path)
            if not df.empty:
                # Find date column
                date_cols = ['issued_date', 'date', 'timestamp', 'Date', 'DATE']
                date_col = next((col for col in date_cols if col in df.columns), None)
                
                if date_col:
                    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                    df.set_index('date', inplace=True)
                else:
                    # Create dates if none found
                    df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
                
                data['daily_stats'] = df
        
        # Load anomaly results
        anomaly_path = os.path.join(project_root, "data", "output", "anomaly_results.csv")
        if os.path.exists(anomaly_path):
            df = pd.read_csv(anomaly_path)
            if not df.empty:
                date_cols = ['issued_date', 'date', 'timestamp']
                date_col = next((col for col in date_cols if col in df.columns), None)
                
                if date_col:
                    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                    df.set_index('date', inplace=True)
                
                # Ensure required columns exist
                if 'is_anomaly' not in df.columns:
                    df['is_anomaly'] = False
                if 'anomaly_score' not in df.columns:
                    df['anomaly_score'] = 0.0
                
                data['anomalies'] = df
        
        # Load forecasts
        forecast_path = os.path.join(project_root, "data", "output", "forecast_results.csv")
        if os.path.exists(forecast_path):
            df = pd.read_csv(forecast_path)
            if not df.empty:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                data['forecasts'] = df
        
        # Load processed alerts
        alert_path = os.path.join(project_root, "data", "processed", "weather_alerts_processed.csv")
        if os.path.exists(alert_path):
            df = pd.read_csv(alert_path)
            if not df.empty:
                data['alerts'] = df
        
        # Load insights from anomaly explanations
        insight_path = os.path.join(project_root, "data", "output", "anomaly_results_explanations.json")
        if os.path.exists(insight_path):
            try:
                with open(insight_path, 'r') as f:
                    insights_data = json.load(f)
                    if isinstance(insights_data, dict):
                        if 'message' in insights_data:
                            data['insights'].append(insights_data['message'])
                        if 'anomalies' in insights_data and insights_data['anomalies']:
                            anomaly_count = len(insights_data['anomalies'])
                            data['insights'].append(f"Detected {anomaly_count} anomalies in the data")
            except:
                pass
        
        # Determine system status
        if not data['daily_stats'].empty:
            data['system_status'] = {
                'data_source': 'Live Data',
                'data_quality': 'high' if len(data['daily_stats']) > 10 else 'medium',
                'last_updated': datetime.fromtimestamp(
                    os.path.getmtime(daily_path) if os.path.exists(daily_path) else time.time()
                ).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            # If no real data, create minimal demo but indicate it's not real
            data = create_fallback_data()
            data['system_status'] = {
                'data_source': 'No Data Available - Run Pipeline First',
                'data_quality': 'low',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)[:200]}")
        # Return fallback data on error
        data = create_fallback_data()
        data['system_status'] = {
            'data_source': 'Error Loading Data',
            'data_quality': 'low',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return data

def create_fallback_data():
    """Create minimal fallback data when real data is not available."""
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    
    daily_stats = pd.DataFrame({
        'date': dates,
        'total_alerts': [0] * len(dates),
        'flood': [0] * len(dates),
        'storm': [0] * len(dates),
        'wind': [0] * len(dates)
    })
    daily_stats.set_index('date', inplace=True)
    
    anomalies = daily_stats.copy()
    anomalies['is_anomaly'] = False
    anomalies['anomaly_score'] = 0.0
    
    forecasts = pd.DataFrame({
        'date': pd.date_range(start=dates[-1] + timedelta(days=1), periods=3, freq='D'),
        'target': 'total_alerts',
        'forecast': [0, 0, 0],
        'lower_bound': [0, 0, 0],
        'upper_bound': [0, 0, 0]
    })
    
    alerts = pd.DataFrame({
        'alert_id': ['NO_DATA_001'],
        'headline': ['No Data Available'],
        'description': ['Run the data collection pipeline to collect weather alerts'],
        'severity': ['Info'],
        'alert_type': ['other'],
        'issued_date': [datetime.now().strftime('%Y-%m-%d')]
    })
    
    insights = [
        "No weather data available. Run the data collection pipeline to collect real weather alerts.",
        "After data collection, run preprocessing and analysis pipelines.",
        "Check the system logs for any errors."
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
            mode='lines+markers',
            name='Total Alerts',
            line=dict(color='#2563eb', width=2),
            marker=dict(size=6)
        ))
    
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly'] == True]
        if not anomaly_points.empty and 'total_alerts' in anomaly_points.columns:
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points['total_alerts'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='#dc2626',
                    size=12,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                )
            ))
    
    fig.update_layout(
        title='Daily Weather Alert Trends',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_forecast_chart(forecasts):
    """Create forecast chart."""
    fig = go.Figure()
    
    if forecasts.empty:
        return fig
    
    if 'target' in forecasts.columns:
        total_forecast = forecasts[forecasts['target'] == 'total_alerts']
    else:
        total_forecast = forecasts
    
    if total_forecast.empty:
        total_forecast = forecasts.head(7)
    
    if not total_forecast.empty and 'forecast' in total_forecast.columns:
        fig.add_trace(go.Scatter(
            x=total_forecast['date'],
            y=total_forecast['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#059669', width=3)
        ))
    
        if 'lower_bound' in total_forecast.columns and 'upper_bound' in total_forecast.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([total_forecast['date'], total_forecast['date'][::-1]]),
                y=pd.concat([total_forecast['upper_bound'], total_forecast['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(5, 150, 105, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Confidence Interval'
            ))
    
    fig.update_layout(
        title='Weather Alert Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Alerts',
        template='plotly_white',
        height=350
    )
    
    return fig

def create_alert_type_chart(daily_stats):
    """Create alert type distribution chart."""
    fig = go.Figure()
    
    alert_type_cols = [col for col in ['flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold'] 
                      if col in daily_stats.columns]
    
    if not alert_type_cols:
        return fig
    
    # Get recent data (last 30 days or all if less)
    recent_data = daily_stats.tail(30) if len(daily_stats) >= 30 else daily_stats
    type_totals = recent_data[alert_type_cols].sum().sort_values(ascending=False)
    
    if not type_totals.empty:
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']
        
        fig.add_trace(go.Bar(
            x=type_totals.index,
            y=type_totals.values,
            marker_color=colors[:len(type_totals)],
            text=type_totals.values,
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Alert Type Distribution (Recent Data)',
        xaxis_title='Alert Type',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=350
    )
    
    return fig

def main():
    """Main dashboard function."""
    st.title("Weather Anomaly Detection System")
    
    # Load data
    data = load_real_data()
    system_status = data.get('system_status', {})
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.header("Data Status")
        
        data_source = system_status.get('data_source', 'Unknown')
        data_quality = system_status.get('data_quality', 'low')
        
        if 'Live' in data_source:
            st.success(f"Data Source: {data_source}")
        elif 'No Data' in data_source or 'Error' in data_source:
            st.error(f"Data Source: {data_source}")
        else:
            st.warning(f"Data Source: {data_source}")
        
        st.write(f"Data Quality: {data_quality}")
        
        if 'last_updated' in system_status:
            st.write(f"Last Updated: {system_status['last_updated']}")
        
        if not data['daily_stats'].empty:
            days = len(data['daily_stats'])
            st.metric("Days of Data", days)
            
            if 'total_alerts' in data['daily_stats'].columns:
                total = data['daily_stats']['total_alerts'].sum()
                st.metric("Total Alerts", int(total))
        
        st.header("Actions")
        st.info("Run data pipeline from main.py to collect and process weather data")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomalies", "Forecasts", "Alert Details"])
    
    with tab1:
        st.header("System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
                total_alerts = data['daily_stats']['total_alerts'].sum()
                st.metric("Total Alerts", f"{int(total_alerts)}")
            else:
                st.metric("Total Alerts", "0")
        
        with col2:
            if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
                avg_daily = data['daily_stats']['total_alerts'].mean()
                st.metric("Avg Daily Alerts", f"{avg_daily:.1f}")
            else:
                st.metric("Avg Daily Alerts", "0.0")
        
        with col3:
            if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
                anomaly_count = data['anomalies']['is_anomaly'].sum()
                st.metric("Detected Anomalies", f"{int(anomaly_count)}")
            else:
                st.metric("Detected Anomalies", "0")
        
        with col4:
            if not data['forecasts'].empty and 'forecast' in data['forecasts'].columns:
                next_forecast = data['forecasts'].iloc[0]['forecast'] if len(data['forecasts']) > 0 else 0
                st.metric("Next Forecast", f"{int(next_forecast)}")
            else:
                st.metric("Next Forecast", "0")
        
        # Charts
        st.subheader("Alert Trends and Anomalies")
        
        timeline_chart = create_timeline_chart(data['daily_stats'], data['anomalies'])
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
        else:
            st.warning("No trend data available")
        
        # Alert type distribution and forecast
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alert Type Distribution")
            type_chart = create_alert_type_chart(data['daily_stats'])
            if type_chart:
                st.plotly_chart(type_chart, use_container_width=True)
            else:
                st.info("No alert type data available")
        
        with col2:
            st.subheader("Forecast")
            forecast_chart = create_forecast_chart(data['forecasts'])
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
            else:
                st.info("No forecast data available")
        
        # Insights
        if data['insights']:
            st.subheader("System Insights")
            for insight in data['insights']:
                st.info(insight)
        else:
            st.subheader("System Status")
            st.info("Run the complete analysis pipeline from main.py to generate insights")
    
    with tab2:
        st.header("Anomaly Analysis")
        
        if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomalies = data['anomalies'][data['anomalies']['is_anomaly'] == True]
            
            if not anomalies.empty:
                st.write(f"**Detected {len(anomalies)} anomalies:**")
                
                for idx, (date, row) in enumerate(anomalies.iterrows()):
                    with st.expander(f"Anomaly detected on {date.strftime('%Y-%m-%d')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Total alerts:** {row.get('total_alerts', 'N/A')}")
                            st.write(f"**Anomaly score:** {row.get('anomaly_score', 0):.3f}")
                        with col2:
                            severity = row.get('anomaly_severity', 'unknown')
                            st.write(f"**Severity:** {severity}")
                            if 'anomaly_confidence' in row:
                                st.write(f"**Confidence:** {row['anomaly_confidence']:.3f}")
            else:
                st.success("No anomalies detected in the current data")
            
            # Anomaly data table
            st.subheader("Recent Anomaly Data")
            display_cols = [col for col in ['total_alerts', 'is_anomaly', 'anomaly_score', 'anomaly_severity', 'anomaly_confidence'] 
                          if col in data['anomalies'].columns]
            if display_cols:
                st.dataframe(data['anomalies'][display_cols].tail(30), use_container_width=True)
        else:
            st.warning("Anomaly detection has not been run or no data is available")
            st.info("Run anomaly detection from main.py option 5 to analyze the data")
    
    with tab3:
        st.header("Weather Forecasts")
        
        if not data['forecasts'].empty:
            st.dataframe(data['forecasts'], use_container_width=True)
            
            # Forecast summary
            st.subheader("Forecast Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'forecast' in data['forecasts'].columns:
                    avg = data['forecasts']['forecast'].mean()
                    st.metric("Average Forecast", f"{avg:.1f}")
            
            with col2:
                if 'forecast' in data['forecasts'].columns:
                    maximum = data['forecasts']['forecast'].max()
                    st.metric("Maximum Forecast", f"{maximum:.1f}")
            
            with col3:
                if 'forecast' in data['forecasts'].columns:
                    minimum = data['forecasts']['forecast'].min()
                    st.metric("Minimum Forecast", f"{minimum:.1f}")
        else:
            st.warning("Forecast data not available")
            st.info("Run forecasting from main.py option 6 to generate predictions")
    
    with tab4:
        st.header("Alert Details")
        
        if not data['alerts'].empty:
            st.dataframe(data['alerts'].head(100), use_container_width=True)
            
            # Alert statistics
            st.subheader("Alert Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'severity' in data['alerts'].columns:
                    severity_counts = data['alerts']['severity'].value_counts()
                    st.write("**Severity Distribution:**")
                    st.dataframe(severity_counts)
            
            with col2:
                if 'alert_type' in data['alerts'].columns:
                    type_counts = data['alerts']['alert_type'].value_counts()
                    st.write("**Alert Type Distribution:**")
                    st.dataframe(type_counts)
        else:
            st.warning("Alert data not available")
            st.info("Run data collection and preprocessing pipelines to load alert data")
    
    # Footer
    st.write("---")
    st.caption(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import time
    main()
