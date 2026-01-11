#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Complete Backend Connected
Direct connection to all src modules
"""

import os
import sys
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

# ============================================================================
# SETUP - CRITICAL: CREATE DIRECTORIES FIRST
# ============================================================================

# Create ALL required directories
required_dirs = [
    "data/raw",
    "data/processed",
    "data/output",
    "models",
    "logs"
]

for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Add src to Python path - ABSOLUTELY NECESSARY
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ============================================================================
# BACKEND MODULE IMPORTS - DIRECT FROM YOUR SRC FOLDER
# ============================================================================

def import_scraping_module():
    """Import and use your scraping module"""
    try:
        from scraping.scrape_weather_alerts import main as scrape_main
        from scraping.scrape_weather_alerts import WeatherAlertScraper
        return scrape_main, WeatherAlertScraper
    except ImportError as e:
        st.error(f"Scraping module import error: {e}")
        return None, None

def import_preprocessing_module():
    """Import and use your preprocessing module"""
    try:
        from preprocessing.preprocess_text import preprocess_pipeline
        from preprocessing.preprocess_text import WeatherAlertPreprocessor
        return preprocess_pipeline, WeatherAlertPreprocessor
    except ImportError as e:
        st.error(f"Preprocessing module import error: {e}")
        return None, None

def import_anomaly_module():
    """Import and use your anomaly detection module"""
    try:
        from ml.anomaly_detection import run_anomaly_detection
        from ml.anomaly_detection import AnomalyDetector
        return run_anomaly_detection, AnomalyDetector
    except ImportError as e:
        st.error(f"Anomaly module import error: {e}")
        return None, None

def import_forecast_module():
    """Import and use your forecasting module"""
    try:
        from ml.forecast_model import run_forecasting
        from ml.forecast_model import AlertForecaster
        return run_forecasting, AlertForecaster
    except ImportError as e:
        st.error(f"Forecast module import error: {e}")
        return None, None

# ============================================================================
# BACKEND PIPELINE FUNCTIONS - USING YOUR MODULES
# ============================================================================

def run_web_scraping():
    """Run your web scraping module"""
    with st.spinner("Running web scraping from weather.gov..."):
        try:
            scrape_main, _ = import_scraping_module()
            if scrape_main is None:
                st.error("Scraping module not found")
                return False
            
            # Run your scraping function
            alert_count = scrape_main()
            
            if alert_count is None:
                st.warning("Scraping returned None, checking for data file")
                if os.path.exists("data/raw/weather_alerts_raw.csv"):
                    df = pd.read_csv("data/raw/weather_alerts_raw.csv")
                    st.success(f"Found existing data: {len(df)} alerts")
                    return True
                else:
                    create_fallback_data()
                    st.warning("Created fallback data for processing")
                    return False
            elif alert_count == 0:
                st.warning("No alerts collected, checking existing data")
                if os.path.exists("data/raw/weather_alerts_raw.csv"):
                    df = pd.read_csv("data/raw/weather_alerts_raw.csv")
                    st.info(f"Using existing data: {len(df)} alerts")
                    return True
                else:
                    create_fallback_data()
                    st.warning("Created fallback data")
                    return False
            else:
                st.success(f"Successfully collected {alert_count} weather alerts")
                return True
                
        except Exception as e:
            st.error(f"Scraping error: {str(e)}")
            create_fallback_data()
            st.info("Created fallback data for processing")
            return False

def create_fallback_data():
    """Create fallback data if scraping fails"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sample_data = []
    alert_types = ['flood', 'storm', 'wind', 'winter', 'heat', 'cold']
    
    for i in range(100):
        alert_date = np.random.choice(dates)
        if hasattr(alert_date, 'strftime'):
            date_str = alert_date.strftime('%Y-%m-%d')
        else:
            date_str = str(alert_date)
            
        alert_type = np.random.choice(alert_types)
        severity = np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.5, 0.3, 0.2])
        
        sample_data.append({
            'alert_id': f'FALLBACK_{i:04d}',
            'headline': f'{severity} {alert_type.title()} Warning',
            'description': f'A {severity.lower()} {alert_type} alert has been issued for testing.',
            'severity': severity,
            'alert_type': alert_type,
            'area': np.random.choice(['Northeast Region', 'Midwest Plains', 'Southwest Desert']),
            'issued_date': date_str,
            'scraped_at': datetime.now().isoformat()
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw/weather_alerts_raw.csv', index=False)
    return True

def run_data_processing():
    """Run your data preprocessing module"""
    with st.spinner("Processing and analyzing weather data..."):
        try:
            preprocess_pipeline, _ = import_preprocessing_module()
            if preprocess_pipeline is None:
                st.error("Preprocessing module not found")
                return False
            
            # Check if raw data exists
            if not os.path.exists("data/raw/weather_alerts_raw.csv"):
                st.error("No raw data found. Run scraping first.")
                return False
            
            # Run your preprocessing function
            processed_df, daily_df = preprocess_pipeline(
                "data/raw/weather_alerts_raw.csv",
                "data/processed/weather_alerts_processed.csv"
            )
            
            if processed_df is not None and not processed_df.empty:
                st.success(f"Processed {len(processed_df)} alerts, created {len(daily_df)} daily records")
                return True
            else:
                st.warning("Processing returned limited data")
                return True
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return False

def run_anomaly_detection():
    """Run your anomaly detection module"""
    with st.spinner("Running anomaly detection on weather patterns..."):
        try:
            run_anomaly_detection_func, _ = import_anomaly_module()
            if run_anomaly_detection_func is None:
                st.error("Anomaly detection module not found")
                return False
            
            # Check if processed data exists
            if not os.path.exists("data/processed/weather_alerts_daily.csv"):
                st.error("No processed data found. Run preprocessing first.")
                return False
            
            # Run your anomaly detection function
            result_df, explanations = run_anomaly_detection_func(
                "data/processed/weather_alerts_daily.csv",
                "data/output/anomaly_results.csv",
                "models/isolation_forest.pkl"
            )
            
            if result_df is not None:
                anomaly_count = result_df['is_anomaly'].sum() if 'is_anomaly' in result_df.columns else 0
                st.success(f"Anomaly detection completed: {anomaly_count} anomalies found")
                return True
            else:
                st.warning("Anomaly detection returned limited results")
                return True
                
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return False

def run_weather_forecasting():
    """Run your forecasting module"""
    with st.spinner("Generating weather forecasts..."):
        try:
            run_forecasting_func, _ = import_forecast_module()
            if run_forecasting_func is None:
                st.error("Forecasting module not found")
                return False
            
            # Check if processed data exists
            if not os.path.exists("data/processed/weather_alerts_daily.csv"):
                st.error("No processed data found. Run preprocessing first.")
                return False
            
            # Run your forecasting function
            forecast_df, status = run_forecasting_func(
                "data/processed/weather_alerts_daily.csv",
                "data/output/forecast_results.csv",
                "models/xgboost_forecast.pkl"
            )
            
            if forecast_df is not None:
                st.success(f"Forecasting completed: {len(forecast_df)} predictions generated")
                return True
            else:
                st.warning("Forecasting returned limited results")
                return True
                
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            return False

# ============================================================================
# STREAMLIT DASHBOARD UI
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
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3B82F6;
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

# Header
st.markdown('<h1 class="main-header">Weather Anomaly Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;">Professional weather monitoring and anomaly detection platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 class="section-header">Backend Pipeline Controls</h3>', unsafe_allow_html=True)
    
    if st.button("Run Complete Pipeline", type="primary", use_container_width=True):
        run_web_scraping()
        run_data_processing()
        run_anomaly_detection()
        run_weather_forecasting()
        st.success("Pipeline execution completed")
    
    st.markdown('<h4 class="section-header">Individual Modules</h4>', unsafe_allow_html=True)
    
    if st.button("Run Web Scraping", use_container_width=True):
        run_web_scraping()
    
    if st.button("Run Data Processing", use_container_width=True):
        run_data_processing()
    
    if st.button("Run Anomaly Detection", use_container_width=True):
        run_anomaly_detection()
    
    if st.button("Run Forecasting", use_container_width=True):
        run_weather_forecasting()
    
    # Module status
    st.markdown('<h4 class="section-header">Module Status</h4>', unsafe_allow_html=True)
    
    modules = [
        ("Web Scraping", import_scraping_module()),
        ("Data Processing", import_preprocessing_module()),
        ("Anomaly Detection", import_anomaly_module()),
        ("Forecasting", import_forecast_module())
    ]
    
    for name, (func, _) in modules:
        if func is not None:
            st.success(f"{name}: Connected")
        else:
            st.error(f"{name}: Not Connected")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Anomaly Results",
    "Forecast Results",
    "System Info"
])

with tab1:
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    
    if os.path.exists("data/processed/weather_alerts_daily.csv"):
        df = pd.read_csv("data/processed/weather_alerts_daily.csv")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_alerts' in df.columns:
                st.metric("Total Days", len(df))
        
        with col2:
            if 'total_alerts' in df.columns:
                st.metric("Total Alerts", int(df['total_alerts'].sum()))
        
        with col3:
            if 'severity_score' in df.columns:
                st.metric("Average Severity", f"{df['severity_score'].mean():.2f}")
        
        # Show data
        st.dataframe(df.tail(20), use_container_width=True)
    else:
        st.info("No processed data available. Run the pipeline first.")

with tab2:
    st.markdown('<h2 class="section-header">Anomaly Detection Results</h2>', unsafe_allow_html=True)
    
    if os.path.exists("data/output/anomaly_results.csv"):
        df = pd.read_csv("data/output/anomaly_results.csv")
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            st.metric("Anomalies Detected", len(anomalies))
            
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
        else:
            st.warning("No anomaly data in results")
    else:
        st.info("Run anomaly detection first")

with tab3:
    st.markdown('<h2 class="section-header">Weather Forecasts</h2>', unsafe_allow_html=True)
    
    if os.path.exists("data/output/forecast_results.csv"):
        df = pd.read_csv("data/output/forecast_results.csv")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run forecasting first")

with tab4:
    st.markdown('<h2 class="section-header">System Information</h2>', unsafe_allow_html=True)
    
    # File status
    st.markdown('<h4>Data Files Status</h4>', unsafe_allow_html=True)
    
    files = [
        ("Raw Data", "data/raw/weather_alerts_raw.csv"),
        ("Processed Data", "data/processed/weather_alerts_daily.csv"),
        ("Anomaly Results", "data/output/anomaly_results.csv"),
        ("Forecast Results", "data/output/forecast_results.csv")
    ]
    
    for name, path in files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.success(f"{name}: {len(df)} records")
            except:
                st.info(f"{name}: Available")
        else:
            st.warning(f"{name}: Not found")
    
    # System info
    st.markdown('<h4>System Details</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"Pandas: {pd.__version__}")
        st.text(f"Streamlit: {st.__version__}")
    
    with col2:
        st.text(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.text(f"Project Root: {PROJECT_ROOT}")
        st.text("Data Source: weather.gov")

# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Check and run initial setup
if not os.path.exists("data/raw/weather_alerts_raw.csv"):
    st.info("First time setup: Running initial data collection...")
    run_web_scraping()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # This ensures Streamlit runs the app
    pass
