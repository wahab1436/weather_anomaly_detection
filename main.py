#!/usr/bin/env python3
"""
Weather Anomaly Detection - Streamlit Cloud Compatible
Direct connection to all backend modules
"""

import os
import sys
import streamlit as st
from pathlib import Path

# ============================================================================
# SETUP - FIRST THING!
# ============================================================================

# Create ALL directories immediately
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

st.set_page_config(
    page_title="Weather Anomaly Detection",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ Weather Anomaly Detection System")
st.markdown("### Live Weather Monitoring & Analysis")

# ============================================================================
# BACKEND FUNCTIONS
# ============================================================================

def run_scraping():
    """Run weather.gov scraping"""
    with st.spinner("Scraping weather data from weather.gov..."):
        try:
            from scraping.scrape_weather_alerts import main as scrape_main
            alert_count = scrape_main()
            
            if alert_count is None or alert_count == 0:
                # Create sample data if scraping fails
                create_sample_data()
                st.warning("Using sample data (scraping returned no data)")
                return False
            else:
                st.success(f"✅ Collected {alert_count} weather alerts")
                return True
                
        except Exception as e:
            st.error(f"❌ Scraping failed: {str(e)}")
            create_sample_data()
            return False

def create_sample_data():
    """Create sample data if scraping fails"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sample_data = []
    alert_types = ['flood', 'storm', 'wind', 'winter']
    
    for i in range(100):
        alert_date = np.random.choice(dates)
        alert_type = np.random.choice(alert_types)
        severity = np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.5, 0.3, 0.2])
        
        sample_data.append({
            'alert_id': f'SAMPLE_{i:04d}',
            'headline': f'{severity} {alert_type.title()} Warning',
            'description': f'A {severity.lower()} {alert_type} alert.',
            'severity': severity,
            'alert_type': alert_type,
            'area': np.random.choice(['Northeast', 'Midwest', 'Southwest']),
            'issued_date': alert_date.strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw/weather_alerts_raw.csv', index=False)

def run_processing():
    """Run data processing"""
    with st.spinner("Processing weather data..."):
        try:
            from preprocessing.preprocess_text import preprocess_pipeline
            
            processed_df, daily_df = preprocess_pipeline(
                "data/raw/weather_alerts_raw.csv",
                "data/processed/weather_alerts_processed.csv"
            )
            
            if processed_df is not None:
                st.success(f"✅ Processed {len(processed_df)} alerts")
                return True
            else:
                st.warning("⚠️ Processing returned no data")
                return False
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return False

def run_anomaly():
    """Run anomaly detection"""
    with st.spinner("Detecting anomalies..."):
        try:
            from ml.anomaly_detection import run_anomaly_detection
            
            result_df, explanations = run_anomaly_detection(
                "data/processed/weather_alerts_daily.csv",
                "data/output/anomaly_results.csv",
                "models/isolation_forest.pkl"
            )
            
            if result_df is not None:
                anomaly_count = result_df['is_anomaly'].sum() if 'is_anomaly' in result_df.columns else 0
                st.success(f"✅ Found {anomaly_count} anomalies")
                return True
            else:
                st.warning("⚠️ Anomaly detection returned no results")
                return False
                
        except Exception as e:
            st.error(f"❌ Anomaly detection failed: {str(e)}")
            return False

def run_forecast():
    """Run forecasting"""
    with st.spinner("Generating forecasts..."):
        try:
            from ml.forecast_model import run_forecasting
            
            forecast_df, status = run_forecasting(
                "data/processed/weather_alerts_daily.csv",
                "data/output/forecast_results.csv",
                "models/xgboost_forecast.pkl"
            )
            
            if forecast_df is not None:
                st.success(f"✅ Generated {len(forecast_df)} forecasts")
                return True
            else:
                st.warning("⚠️ Forecasting returned no results")
                return False
                
        except Exception as e:
            st.error(f"❌ Forecasting failed: {str(e)}")
            return False

# ============================================================================
# MAIN DASHBOARD UI
# ============================================================================

# Sidebar controls
with st.sidebar:
    st.header("🚀 System Controls")
    
    if st.button("▶️ Run Complete Pipeline", type="primary", use_container_width=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            run_scraping()
        with col2:
            run_processing()
        with col3:
            run_anomaly()
        with col4:
            run_forecast()
        
        st.balloons()
    
    st.divider()
    
    st.header("⚙️ Individual Steps")
    if st.button("🌐 Web Scraping", use_container_width=True):
        run_scraping()
    
    if st.button("🔧 Data Processing", use_container_width=True):
        run_processing()
    
    if st.button("🔍 Anomaly Detection", use_container_width=True):
        run_anomaly()
    
    if st.button("📈 Forecasting", use_container_width=True):
        run_forecast()
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    
    status_items = [
        ("Raw Data", "data/raw/weather_alerts_raw.csv"),
        ("Processed Data", "data/processed/weather_alerts_daily.csv"),
        ("Anomaly Results", "data/output/anomaly_results.csv"),
        ("Forecast Results", "data/output/forecast_results.csv")
    ]
    
    for name, path in status_items:
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_csv(path)
                st.success(f"✓ {name}: {len(df)} records")
            except:
                st.info(f"✓ {name}: Ready")
        else:
            st.warning(f"✗ {name}: Not found")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Anomalies", "📈 Forecasts", "⚙️ Configuration"])

with tab1:
    st.header("Weather Alert Dashboard")
    
    # Show data if exists
    if os.path.exists("data/processed/weather_alerts_daily.csv"):
        import pandas as pd
        import plotly.express as px
        
        df = pd.read_csv("data/processed/weather_alerts_daily.csv")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            if 'total_alerts' in df.columns:
                st.metric("Total Alerts", int(df['total_alerts'].sum()))
        with col3:
            if 'severity_score' in df.columns:
                st.metric("Avg Severity", f"{df['severity_score'].mean():.2f}")
        
        # Plot
        if 'total_alerts' in df.columns and 'issued_date' in df.columns:
            fig = px.line(df, x='issued_date', y='total_alerts', 
                         title="Daily Weather Alerts")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available. Run the pipeline first.")

with tab2:
    st.header("Anomaly Detection")
    
    if os.path.exists("data/output/anomaly_results.csv"):
        import pandas as pd
        
        df = pd.read_csv("data/output/anomaly_results.csv")
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            st.metric("Detected Anomalies", len(anomalies))
            
            if not anomalies.empty:
                st.dataframe(anomalies.head(10), use_container_width=True)
        else:
            st.warning("No anomaly data found")
    else:
        st.info("Run anomaly detection first")

with tab3:
    st.header("Weather Forecasts")
    
    if os.path.exists("data/output/forecast_results.csv"):
        import pandas as pd
        import plotly.express as px
        
        df = pd.read_csv("data/output/forecast_results.csv")
        
        st.dataframe(df, use_container_width=True)
        
        # Plot forecasts
        if 'forecast' in df.columns and 'date' in df.columns:
            fig = px.line(df, x='date', y='forecast', 
                         title="7-Day Weather Alert Forecast")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run forecasting first")

with tab4:
    st.header("System Configuration")
    
    st.code(f"""
    Project Root: {PROJECT_ROOT}
    Python: {sys.version}
    
    Directories:
    - data/raw: {os.path.exists('data/raw')}
    - data/processed: {os.path.exists('data/processed')}
    - data/output: {os.path.exists('data/output')}
    - models: {os.path.exists('models')}
    """)
    
    # Module check
    st.subheader("Module Status")
    
    modules = [
        ("Scraping", "scraping.scrape_weather_alerts"),
        ("Preprocessing", "preprocessing.preprocess_text"),
        ("Anomaly Detection", "ml.anomaly_detection"),
        ("Forecasting", "ml.forecast_model")
    ]
    
    for name, module_path in modules:
        try:
            __import__(module_path)
            st.success(f"✓ {name}: Available")
        except ImportError:
            st.error(f"✗ {name}: Not available")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Weather Anomaly Detection System | Live Weather Monitoring</p>
    <p>Data Source: weather.gov | Last Updated: {}</p>
</div>
""".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

# ============================================================================
# AUTO-RUN ON STARTUP
# ============================================================================

# Check if we need to run initial setup
if not os.path.exists("data/raw/weather_alerts_raw.csv"):
    st.info("⚙️ First-time setup: Running initial data collection...")
    run_scraping()
