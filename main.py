"""
Main Weather Anomaly Detection Dashboard - Connected to Backend
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import time
import importlib  # <-- fix for dynamic imports

# Add src to path for backend imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set page config - NO EMOJIS
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
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
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.5rem;
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
    .success-card {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .warning-card {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .error-card {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_backend_data():
    """Load data from backend processing pipeline."""
    data = {
        'daily_stats': pd.DataFrame(),
        'anomalies': pd.DataFrame(),
        'forecasts': pd.DataFrame(),
        'alerts': pd.DataFrame(),
        'insights': []
    }
    
    try:
        # Load daily stats
        if os.path.exists("data/processed/weather_alerts_daily.csv"):
            df = pd.read_csv("data/processed/weather_alerts_daily.csv")
            if 'issued_date' in df.columns:
                df['date'] = pd.to_datetime(df['issued_date'])
                df.set_index('date', inplace=True)
            data['daily_stats'] = df
        
        # Load anomalies
        if os.path.exists("data/output/anomaly_results.csv"):
            df = pd.read_csv("data/output/anomaly_results.csv")
            if 'issued_date' in df.columns:
                df['date'] = pd.to_datetime(df['issued_date'])
                df.set_index('date', inplace=True)
            data['anomalies'] = df
        
        # Load forecasts
        if os.path.exists("data/output/forecast_results.csv"):
            df = pd.read_csv("data/output/forecast_results.csv")
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            data['forecasts'] = df
        
        # Load alerts
        if os.path.exists("data/processed/weather_alerts_processed.csv"):
            df = pd.read_csv("data/processed/weather_alerts_processed.csv")
            if 'issued_date' in df.columns:
                df['issued_date'] = pd.to_datetime(df['issued_date'])
            data['alerts'] = df
        
        # Load insights
        if os.path.exists("data/output/insights.json"):
            with open("data/output/insights.json", 'r') as f:
                insights_data = json.load(f)
                data['insights'] = insights_data.get('insights', [])
        
        return data, "Live Data"
        
    except Exception as e:
        st.warning(f"Backend data loading issue: {e}")
        
        # Sample fallback data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        sample_daily = pd.DataFrame({
            'date': dates,
            'total_alerts': np.random.randint(10, 50, 30),
            'flood': np.random.randint(0, 15, 30),
            'storm': np.random.randint(0, 20, 30),
            'wind': np.random.randint(0, 10, 30),
            'winter': np.random.randint(0, 8, 30),
            'severity_score': np.random.uniform(0.1, 1.0, 30),
            '7_day_avg': np.random.randint(15, 35, 30)
        })
        sample_daily.set_index('date', inplace=True)
        
        sample_anomalies = sample_daily.copy()
        sample_anomalies['is_anomaly'] = False
        anomaly_indices = np.random.choice(range(30), 3, replace=False)
        sample_anomalies.iloc[anomaly_indices, sample_anomalies.columns.get_loc('is_anomaly')] = True
        
        forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7, freq='D')
        sample_forecasts = pd.DataFrame({
            'date': forecast_dates,
            'target': 'total_alerts',
            'forecast': np.random.randint(10, 40, 7),
            'lower_bound': np.random.randint(5, 35, 7),
            'upper_bound': np.random.randint(15, 45, 7)
        })
        
        sample_data = {
            'daily_stats': sample_daily,
            'anomalies': sample_anomalies,
            'forecasts': sample_forecasts,
            'alerts': pd.DataFrame(),
            'insights': [
                "System is collecting initial data. Run data collection pipeline.",
                "No anomalies detected in the sample dataset.",
                "Forecast models require historical data for accurate predictions.",
                "Connect to weather.gov for real-time alert monitoring."
            ]
        }
        
        return sample_data, "Sample Data (Demo Mode)"

# -------------------------------
# Backend Pipeline Runner
# -------------------------------
def run_backend_pipeline(pipeline_type):
    """Run specific backend pipeline."""
    try:
        if pipeline_type == "scraping":
            from scraping.scrape_weather_alerts import main as scrape_main
            with st.spinner("Collecting weather alerts from weather.gov..."):
                scrape_main()
            st.success("Data scraping completed!")
            return True
            
        elif pipeline_type == "preprocessing":
            from preprocessing.preprocess_text import preprocess_pipeline
            with st.spinner("Processing and cleaning alert data..."):
                preprocess_pipeline(
                    "data/raw/weather_alerts_raw.csv",
                    "data/processed/weather_alerts_processed.csv"
                )
            st.success("Data preprocessing completed!")
            return True
            
        elif pipeline_type == "anomaly_detection":
            from ml.anomaly_detection import run_anomaly_detection
            with st.spinner("Running anomaly detection..."):
                run_anomaly_detection(
                    "data/processed/weather_alerts_daily.csv",
                    "data/output/anomaly_results.csv",
                    "models/isolation_forest.pkl"
                )
            st.success("Anomaly detection completed!")
            return True
            
        elif pipeline_type == "forecasting":
            from ml.forecast_model import run_forecasting
            with st.spinner("Running forecast models..."):
                run_forecasting(
                    "data/processed/weather_alerts_daily.csv",
                    "data/output/forecast_results.csv",
                    "models/xgboost_forecast.pkl"
                )
            st.success("Forecast generation completed!")
            return True
            
        elif pipeline_type == "complete":
            from scraping.scrape_weather_alerts import main as scrape_main
            from preprocessing.preprocess_text import preprocess_pipeline
            from ml.anomaly_detection import run_anomaly_detection
            from ml.forecast_model import run_forecasting
            
            progress = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/4: Collecting weather alerts...")
            scrape_main()
            progress.progress(25)
            time.sleep(1)
            
            status_text.text("Step 2/4: Processing data...")
            preprocess_pipeline(
                "data/raw/weather_alerts_raw.csv",
                "data/processed/weather_alerts_processed.csv"
            )
            progress.progress(50)
            time.sleep(1)
            
            status_text.text("Step 3/4: Detecting anomalies...")
            run_anomaly_detection(
                "data/processed/weather_alerts_daily.csv",
                "data/output/anomaly_results.csv",
                "models/isolation_forest.pkl"
            )
            progress.progress(75)
            time.sleep(1)
            
            status_text.text("Step 4/4: Generating forecasts...")
            run_forecasting(
                "data/processed/weather_alerts_daily.csv",
                "data/output/forecast_results.csv",
                "models/xgboost_forecast.pkl"
            )
            progress.progress(100)
            status_text.text("Complete!")
            
            st.success("Complete pipeline executed successfully!")
            return True
            
    except Exception as e:
        st.error(f"Error running {pipeline_type} pipeline: {str(e)}")
        return False

# -------------------------------
# Dashboard Charts
# -------------------------------
def create_alert_timeline(daily_stats, anomalies):
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if not daily_stats.empty and 'total_alerts' in daily_stats.columns:
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['total_alerts'],
            mode='lines',
            name='Total Alerts',
            line=dict(color='#3B82F6', width=2)
        ))
    
    if not daily_stats.empty and '7_day_avg' in daily_stats.columns:
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['7_day_avg'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#6B7280', width=1, dash='dash')
        ))
    
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly']]
        if not anomaly_points.empty and 'total_alerts' in anomaly_points.columns:
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points['total_alerts'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='#DC2626',
                    size=10,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                )
            ))
    
    fig.update_layout(
        title='Daily Weather Alerts with Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_alert_type_chart(daily_stats):
    import plotly.graph_objects as go
    
    alert_type_cols = [col for col in daily_stats.columns if col in [
        'flood', 'storm', 'wind', 'winter', 'fire', 
        'heat', 'cold', 'coastal', 'air', 'other'
    ]]
    
    if not alert_type_cols or daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alert type data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    recent_data = daily_stats.tail(30) if len(daily_stats) >= 30 else daily_stats
    type_totals = recent_data[alert_type_cols].sum().sort_values(ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=type_totals.index,
            y=type_totals.values,
            marker_color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
                         '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1'],
            text=type_totals.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Alert Type Distribution (Last 30 Days)',
        xaxis_title='Alert Type',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=300
    )
    
    return fig

def create_forecast_chart(forecasts):
    import plotly.graph_objects as go
    
    if forecasts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    total_forecast = forecasts[forecasts['target'] == 'total_alerts']
    if total_forecast.empty:
        total_forecast = forecasts.head(7)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=total_forecast['date'],
        y=total_forecast['forecast'],
        mode='lines',
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
            name='Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title='7-Day Alert Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Alerts',
        template='plotly_white',
        height=300,
        hovermode='x unified'
    )
    
    return fig

# -------------------------------
# Main Dashboard Function
# -------------------------------
def main():
    st.markdown('<h1 class="main-header">Weather Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    st.write("### Professional Weather Alert Monitoring System - Connected Backend")
    
    data, data_source = load_backend_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Backend Controls")
        
        if st.button("Run Complete Pipeline"):
            if run_backend_pipeline("complete"):
                st.cache_data.clear()
                st.rerun()
        
        if st.button("Refresh Dashboard"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("### Individual Pipeline Steps")
        
        pipeline_steps = [
            ("Collect Data", "scraping"),
            ("Process Data", "preprocessing"),
            ("Detect Anomalies", "anomaly_detection"),
            ("Generate Forecasts", "forecasting")
        ]
        
        for step_name, step_key in pipeline_steps:
            if st.button(f"Run {step_name}"):
                if run_backend_pipeline(step_key):
                    st.cache_data.clear()
                    st.rerun()
        
        # Data source info
        st.markdown("---")
        st.markdown(f"**Data Source:** {data_source}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Anomalies", "Forecasts", "Alerts", "Backend Status"])
    
    # -------------------------------
    # Backend Status Fix (dynamic import safe)
    # -------------------------------
    with tab5:
        st.markdown('<h2 class="sub-header">Backend System Status</h2>', unsafe_allow_html=True)
        st.markdown("### Backend Modules Status")
        
        modules = [
            ("Web Scraping", "scraping.scrape_weather_alerts"),
            ("Data Preprocessing", "preprocessing.preprocess_text"),
            ("Anomaly Detection", "ml.anomaly_detection"),
            ("Forecast Models", "ml.forecast_model"),
            ("Utilities", "utils.helpers")
        ]
        
        for module_name, module_path in modules:
            try:
                importlib.import_module(module_path)
                st.markdown(f"✓ **{module_name}**: Available")
            except Exception:
                st.markdown(f"✗ **{module_name}**: Not Available")
    
    # Other tabs remain unchanged (Overview, Anomalies, Forecasts, Alerts)
    # ... your existing tab logic can remain here
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Weather Anomaly Detection Dashboard v1.0 | Connected Backend System</p>
        <p>Data Source: {data_source} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>National Weather Service Integration | Production-Ready Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Scheduler Wrapper
# -------------------------------
class WeatherAnomalySystem:
    def __init__(self):
        pass

    def run_complete_pipeline(self):
        from main import run_backend_pipeline
        return run_backend_pipeline("complete")

    def run_pipeline_step(self, step_name):
        from main import run_backend_pipeline
        return run_backend_pipeline(step_name)

    def run_scheduler(self):
        import json
        with open("config.json", "r") as f:
            config = json.load(f)
        ml_interval = config.get("ml_interval_hours", 6) * 3600
        while True:
            print(f"[{datetime.now()}] Running scheduled complete pipeline...")
            self.run_complete_pipeline()
            print(f"[{datetime.now()}] Pipeline finished. Sleeping {ml_interval} seconds...")
            time.sleep(ml_interval)

# -------------------------------
# Run dashboard
# -------------------------------
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist
