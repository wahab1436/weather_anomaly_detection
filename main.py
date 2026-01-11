#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Professional Dashboard
No emojis, clean professional interface
"""

import os
import sys
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

# ============================================================================
# SETUP
# ============================================================================

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================

st.set_page_config(
    page_title="Weather Anomaly Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BACKEND FUNCTIONS
# ============================================================================

def create_sample_data():
    """Create sample data for testing"""
    import numpy as np
    from datetime import datetime
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sample_data = []
    alert_types = ['flood', 'storm', 'wind', 'winter']
    
    for i in range(100):
        alert_date = np.random.choice(dates)
        if isinstance(alert_date, pd.Timestamp):
            date_str = alert_date.strftime('%Y-%m-%d')
        else:
            date_str = str(alert_date)
            
        alert_type = np.random.choice(alert_types)
        severity = np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.5, 0.3, 0.2])
        
        sample_data.append({
            'alert_id': f'SAMPLE_{i:04d}',
            'headline': f'{severity} {alert_type.title()} Warning',
            'description': f'A {severity.lower()} {alert_type} alert.',
            'severity': severity,
            'alert_type': alert_type,
            'area': np.random.choice(['Northeast', 'Midwest', 'Southwest']),
            'issued_date': date_str
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw/weather_alerts_raw.csv', index=False)
    return True

def run_scraping():
    """Run web scraping"""
    with st.spinner("Collecting weather data from sources..."):
        try:
            from scraping.scrape_weather_alerts import main as scrape_main
            alert_count = scrape_main()
            
            if alert_count is None or alert_count == 0:
                create_sample_data()
                st.warning("Using sample data (scraping returned no data)")
                return False
            else:
                st.success(f"Collected {alert_count} weather alerts")
                return True
                
        except Exception as e:
            st.error(f"Scraping failed: {str(e)}")
            create_sample_data()
            return False

def run_processing():
    """Run data processing"""
    with st.spinner("Processing and analyzing weather data..."):
        try:
            from preprocessing.preprocess_text import preprocess_pipeline
            
            processed_df, daily_df = preprocess_pipeline(
                "data/raw/weather_alerts_raw.csv",
                "data/processed/weather_alerts_processed.csv"
            )
            
            if processed_df is not None:
                st.success(f"Processed {len(processed_df)} alerts")
                return True
            else:
                st.warning("Processing returned no data")
                return False
                
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            return False

def run_anomaly():
    """Run anomaly detection"""
    with st.spinner("Analyzing patterns and detecting anomalies..."):
        try:
            from ml.anomaly_detection import run_anomaly_detection
            
            result_df, explanations = run_anomaly_detection(
                "data/processed/weather_alerts_daily.csv",
                "data/output/anomaly_results.csv",
                "models/isolation_forest.pkl"
            )
            
            if result_df is not None:
                anomaly_count = result_df['is_anomaly'].sum() if 'is_anomaly' in result_df.columns else 0
                st.success(f"Found {anomaly_count} anomalies")
                return True
            else:
                st.warning("Anomaly detection returned no results")
                return False
                
        except Exception as e:
            st.error(f"Anomaly detection failed: {str(e)}")
            return False

def run_forecast():
    """Run forecasting"""
    with st.spinner("Generating weather forecasts..."):
        try:
            from ml.forecast_model import run_forecasting
            
            forecast_df, status = run_forecasting(
                "data/processed/weather_alerts_daily.csv",
                "data/output/forecast_results.csv",
                "models/xgboost_forecast.pkl"
            )
            
            if forecast_df is not None:
                st.success(f"Generated {len(forecast_df)} forecasts")
                return True
            else:
                st.warning("Forecasting returned no results")
                return False
                
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")
            return False

# ============================================================================
# DASHBOARD UI
# ============================================================================

# Header
st.markdown('<h1 class="main-header">Weather Anomaly Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;">Professional weather monitoring and anomaly detection platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 class="section-header">System Controls</h3>', unsafe_allow_html=True)
    
    if st.button("Run Complete Analysis Pipeline", type="primary", use_container_width=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            run_scraping()
        with col2:
            run_processing()
        with col3:
            run_anomaly()
        with col4:
            run_forecast()
    
    st.markdown('<h4 class="section-header">Pipeline Components</h4>', unsafe_allow_html=True)
    
    if st.button("Collect Weather Data", use_container_width=True):
        run_scraping()
    
    if st.button("Process & Analyze Data", use_container_width=True):
        run_processing()
    
    if st.button("Detect Anomalies", use_container_width=True):
        run_anomaly()
    
    if st.button("Generate Forecasts", use_container_width=True):
        run_forecast()
    
    # System status
    st.markdown('<h4 class="section-header">System Status</h4>', unsafe_allow_html=True)
    
    status_items = [
        ("Raw Data", "data/raw/weather_alerts_raw.csv"),
        ("Processed Data", "data/processed/weather_alerts_daily.csv"),
        ("Anomaly Results", "data/output/anomaly_results.csv"),
        ("Forecast Results", "data/output/forecast_results.csv")
    ]
    
    for name, path in status_items:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.markdown(f"**{name}:** {len(df)} records")
            except:
                st.markdown(f"**{name}:** Available")
        else:
            st.markdown(f"**{name}:** Not available")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard Overview",
    "Anomaly Analysis", 
    "Forecasts",
    "System Configuration"
])

with tab1:
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if os.path.exists("data/processed/weather_alerts_daily.csv"):
        df = pd.read_csv("data/processed/weather_alerts_daily.csv")
        
        with col1:
            if 'total_alerts' in df.columns:
                total = df['total_alerts'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Alerts</h3>
                    <div style="font-size: 2.25rem; font-weight: 700;">{int(total)}</div>
                    <div style="font-size: 0.875rem; color: #6B7280;">Across all time periods</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'severity_score' in df.columns:
                avg = df['severity_score'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Average Severity</h3>
                    <div style="font-size: 2.25rem; font-weight: 700;">{avg:.2f}</div>
                    <div style="font-size: 0.875rem; color: #6B7280;">0.0 - 1.0 scale</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Charts
    if os.path.exists("data/processed/weather_alerts_daily.csv"):
        df = pd.read_csv("data/processed/weather_alerts_daily.csv")
        
        if 'total_alerts' in df.columns and 'issued_date' in df.columns:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['issued_date'],
                y=df['total_alerts'],
                mode='lines',
                name='Total Alerts',
                line=dict(color='#3B82F6', width=3)
            ))
            
            fig.update_layout(
                title='Daily Weather Alert Trends',
                xaxis_title='Date',
                yaxis_title='Number of Alerts',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<h2 class="section-header">Anomaly Analysis</h2>', unsafe_allow_html=True)
    
    if os.path.exists("data/output/anomaly_results.csv"):
        df = pd.read_csv("data/output/anomaly_results.csv")
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            st.metric("Detected Anomalies", len(anomalies))
            
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
    else:
        st.info("Run anomaly detection to see results")

with tab3:
    st.markdown('<h2 class="section-header">Weather Forecasts</h2>', unsafe_allow_html=True)
    
    if os.path.exists("data/output/forecast_results.csv"):
        df = pd.read_csv("data/output/forecast_results.csv")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run forecasting to see predictions")

with tab4:
    st.markdown('<h2 class="section-header">System Configuration</h2>', unsafe_allow_html=True)
    
    # Module status
    st.markdown('<h4>Module Status</h4>', unsafe_allow_html=True)
    
    modules = [
        ("Data Collection", "scraping.scrape_weather_alerts"),
        ("Data Processing", "preprocessing.preprocess_text"),
        ("Anomaly Detection", "ml.anomaly_detection"),
        ("Forecasting", "ml.forecast_model")
    ]
    
    for name, module_path in modules:
        try:
            __import__(module_path)
            st.success(f"{name}: Available")
        except ImportError:
            st.error(f"{name}: Not available")
    
    # System info
    st.markdown('<h4>System Information</h4>', unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        **Python Version:** {sys.version.split()[0]}
        
        **Pandas Version:** {pd.__version__}
        """)
    
    with info_col2:
        st.markdown(f"""
        **Streamlit Version:** {st.__version__}
        
        **Project Root:** {PROJECT_ROOT}
        
        **Data Source:** weather.gov
        """)

# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Run initial scraping on first load
if not os.path.exists("data/raw/weather_alerts_raw.csv"):
    st.info("Initializing system: Collecting weather data...")
    run_scraping()
