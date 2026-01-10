#!/usr/bin/env python3
"""
Weather Anomaly Detection Dashboard - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json

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
def load_data():
    """Load data from backend processing."""
    data = {
        'daily_stats': pd.DataFrame(),
        'anomalies': pd.DataFrame(),
        'forecasts': pd.DataFrame(),
        'alerts': pd.DataFrame(),
        'insights': []
    }
    
    # Try to load from data files
    data_files = {
        'daily_stats': os.path.join(project_root, 'data', 'processed', 'weather_alerts_daily.csv'),
        'anomalies': os.path.join(project_root, 'data', 'output', 'anomaly_results.csv'),
        'forecasts': os.path.join(project_root, 'data', 'output', 'forecast_results.csv'),
        'alerts': os.path.join(project_root, 'data', 'processed', 'weather_alerts_processed.csv')
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
        os.path.join(project_root, 'data', 'output', 'anomaly_results_explanations.json'),
        os.path.join(project_root, 'data', 'output', 'insights.json')
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
    
    return data

def main():
    """Main dashboard function."""
    st.title("Weather Anomaly Detection System")
    
    # Load data
    data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.header("Data Status")
        
        if not data['daily_stats'].empty:
            days = len(data['daily_stats'])
            st.write(f"Days of data: {days}")
            
            if 'total_alerts' in data['daily_stats'].columns:
                avg_alerts = data['daily_stats']['total_alerts'].mean()
                st.write(f"Average daily alerts: {avg_alerts:.1f}")
        else:
            st.warning("No data available. Run the pipeline first.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomalies", "Forecasts", "Alerts"])
    
    with tab1:
        st.header("System Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
                total = data['daily_stats']['total_alerts'].sum()
                st.metric("Total Alerts", f"{int(total)}")
            else:
                st.metric("Total Alerts", "0")
        
        with col2:
            if not data['daily_stats'].empty and 'severity_score' in data['daily_stats'].columns:
                avg_severity = data['daily_stats']['severity_score'].mean()
                st.metric("Avg Severity", f"{avg_severity:.2f}")
            else:
                st.metric("Avg Severity", "0.0")
        
        with col3:
            if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
                anomaly_count = data['anomalies']['is_anomaly'].sum()
                st.metric("Anomalies", f"{int(anomaly_count)}")
            else:
                st.metric("Anomalies", "0")
        
        with col4:
            if not data['forecasts'].empty:
                if 'forecast' in data['forecasts'].columns:
                    next_forecast = data['forecasts'].iloc[0]['forecast'] if len(data['forecasts']) > 0 else 0
                    st.metric("Next Forecast", f"{int(next_forecast)}")
                else:
                    st.metric("Next Forecast", "0")
            else:
                st.metric("Next Forecast", "0")
        
        # Charts
        st.subheader("Alert Trends")
        
        if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
            st.line_chart(data['daily_stats']['total_alerts'])
        else:
            st.info("No trend data available")
        
        # Alert type distribution
        st.subheader("Alert Type Distribution")
        
        if not data['daily_stats'].empty:
            alert_cols = [col for col in ['flood', 'storm', 'wind', 'winter', 'fire', 'heat'] 
                         if col in data['daily_stats'].columns]
            if alert_cols:
                recent = data['daily_stats'][alert_cols].tail(7).sum()
                st.bar_chart(recent)
            else:
                st.info("No alert type data available")
        else:
            st.info("No data available")
    
    with tab2:
        st.header("Anomaly Analysis")
        
        if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomalies = data['anomalies'][data['anomalies']['is_anomaly'] == True]
            
            if not anomalies.empty:
                st.write(f"Found {len(anomalies)} anomalies:")
                
                for idx, (date, row) in enumerate(anomalies.iterrows()):
                    with st.expander(f"Anomaly detected on {date.strftime('%Y-%m-%d')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Total alerts: {row.get('total_alerts', 'N/A')}")
                            st.write(f"Anomaly score: {row.get('anomaly_score', 0):.3f}")
                        with col2:
                            st.write(f"Severity: {row.get('anomaly_severity', 'low')}")
                            st.write(f"Confidence: {row.get('anomaly_confidence', 0):.3f}")
            else:
                st.success("No anomalies detected")
            
            # Data table
            st.subheader("Anomaly Data")
            display_cols = [col for col in ['total_alerts', 'is_anomaly', 'anomaly_score', 'anomaly_severity'] 
                          if col in data['anomalies'].columns]
            if display_cols:
                st.dataframe(data['anomalies'][display_cols].tail(30))
        else:
            st.warning("Anomaly data not available")
    
    with tab3:
        st.header("Forecast Analysis")
        
        if not data['forecasts'].empty:
            st.dataframe(data['forecasts'])
            
            # Forecast metrics
            col1, col2 = st.columns(2)
            with col1:
                if 'forecast' in data['forecasts'].columns:
                    avg = data['forecasts']['forecast'].mean()
                    st.metric("Average", f"{avg:.1f}")
            with col2:
                if 'forecast' in data['forecasts'].columns:
                    maximum = data['forecasts']['forecast'].max()
                    st.metric("Maximum", f"{maximum:.1f}")
        else:
            st.warning("Forecast data not available")
    
    with tab4:
        st.header("Alert Details")
        
        if not data['alerts'].empty:
            st.dataframe(data['alerts'].head(50))
            
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                if 'severity' in data['alerts'].columns:
                    severity_counts = data['alerts']['severity'].value_counts()
                    st.write("Severity Distribution:")
                    st.write(severity_counts)
            with col2:
                if 'alert_type' in data['alerts'].columns:
                    type_counts = data['alerts']['alert_type'].value_counts()
                    st.write("Alert Type Distribution:")
                    st.write(type_counts)
        else:
            st.warning("Alert data not available")
    
    # Footer
    st.write("---")
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
