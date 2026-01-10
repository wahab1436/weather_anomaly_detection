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
import importlib

# Set page config
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
        daily_paths = [
            "data/processed/weather_alerts_daily.csv",
            "data/processed/daily_alerts.csv",
            "data/output/daily_stats.csv"
        ]
        
        for path in daily_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    # Try different date column names
                    date_columns = ['issued_date', 'date', 'timestamp', 'Date', 'DATE']
                    date_col = None
                    for col in date_columns:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                        df.set_index('date', inplace=True)
                    else:
                        # Create date index if no date column
                        df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
                    
                    data['daily_stats'] = df
                    break
        
        # Load anomalies
        anomaly_paths = [
            "data/output/anomaly_results.csv",
            "data/processed/anomaly_detection.csv",
            "data/anomalies.csv"
        ]
        
        for path in anomaly_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    # Try different date column names
                    date_columns = ['issued_date', 'date', 'timestamp', 'Date', 'DATE']
                    date_col = None
                    for col in date_columns:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                        df.set_index('date', inplace=True)
                    
                    # Ensure required columns exist
                    if 'is_anomaly' not in df.columns:
                        df['is_anomaly'] = False
                    
                    data['anomalies'] = df
                    break
        
        # Load forecasts
        forecast_paths = [
            "data/output/forecast_results.csv",
            "data/forecasts.csv",
            "data/processed/forecasts.csv"
        ]
        
        for path in forecast_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    data['forecasts'] = df
                    break
        
        # Load processed alerts
        alert_paths = [
            "data/processed/weather_alerts_processed.csv",
            "data/raw/weather_alerts_raw.csv",
            "data/alerts.csv"
        ]
        
        for path in alert_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    # Try to get date column
                    date_columns = ['issued_date', 'date', 'timestamp', 'Date', 'DATE']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            break
                    data['alerts'] = df
                    break
        
        # Load insights if available
        insight_paths = [
            "data/output/anomaly_results_explanations.json",
            "data/output/insights.json",
            "data/insights.json"
        ]
        
        for path in insight_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        insights_data = json.load(f)
                        if isinstance(insights_data, dict):
                            data['insights'] = insights_data.get('insights', [])
                            if not data['insights'] and 'message' in insights_data:
                                data['insights'] = [insights_data['message']]
                        elif isinstance(insights_data, list):
                            data['insights'] = insights_data
                except:
                    pass
        
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

def run_backend_pipeline(pipeline_type):
    """Run specific backend pipeline."""
    try:
        if pipeline_type == "scraping":
            try:
                from scraping.scrape_weather_alerts import main as scrape_main
                with st.spinner("Collecting weather alerts from weather.gov..."):
                    result = scrape_main()
                    if result:
                        st.success("Data scraping completed!")
                        return True
                    else:
                        st.warning("Scraping completed but may have encountered issues.")
                        return True
            except Exception as e:
                st.error(f"Scraping error: {str(e)}")
                return False
            
        elif pipeline_type == "preprocessing":
            try:
                from preprocessing.preprocess_text import preprocess_pipeline
                with st.spinner("Processing and cleaning alert data..."):
                    preprocess_pipeline(
                        "data/raw/weather_alerts_raw.csv",
                        "data/processed/weather_alerts_processed.csv"
                    )
                st.success("Data preprocessing completed!")
                return True
            except Exception as e:
                st.error(f"Preprocessing error: {str(e)}")
                return False
            
        elif pipeline_type == "anomaly_detection":
            try:
                from ml.anomaly_detection import run_anomaly_detection
                with st.spinner("Running anomaly detection..."):
                    run_anomaly_detection(
                        "data/processed/weather_alerts_daily.csv",
                        "data/output/anomaly_results.csv",
                        "models/isolation_forest.pkl"
                    )
                st.success("Anomaly detection completed!")
                return True
            except Exception as e:
                st.error(f"Anomaly detection error: {str(e)}")
                return False
            
        elif pipeline_type == "forecasting":
            try:
                from ml.forecast_model import run_forecasting
                with st.spinner("Running forecast models..."):
                    result_df, status = run_forecasting(
                        "data/processed/weather_alerts_daily.csv",
                        "data/output/forecast_results.csv",
                        "models/xgboost_forecast.pkl"
                    )
                    if not result_df.empty:
                        st.success(f"Forecast generation completed! Generated {len(result_df)} forecasts.")
                    else:
                        st.warning("Forecast generation completed but no forecasts were generated.")
                return True
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                return False
            
        elif pipeline_type == "complete":
            progress = st.progress(0)
            status_text = st.empty()
            
            steps = [
                ("Collecting weather alerts...", "scraping"),
                ("Processing data...", "preprocessing"),
                ("Detecting anomalies...", "anomaly_detection"),
                ("Generating forecasts...", "forecasting")
            ]
            
            for i, (message, step_type) in enumerate(steps):
                status_text.text(f"Step {i+1}/4: {message}")
                success = run_backend_pipeline(step_type)
                if not success:
                    st.error(f"Pipeline failed at step {i+1}: {message}")
                    return False
                progress.progress((i + 1) * 25)
            
            status_text.text("Complete!")
            st.success("Complete pipeline executed successfully!")
            return True
            
    except Exception as e:
        st.error(f"Error running {pipeline_type} pipeline: {str(e)}")
        return False

def create_alert_timeline(daily_stats, anomalies):
    """Create timeline chart of alerts."""
    try:
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
            anomaly_points = anomalies[anomalies['is_anomaly'] == True]
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
    except:
        return None

def create_alert_type_chart(daily_stats):
    """Create alert type distribution chart."""
    try:
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
    except:
        return None

def create_forecast_chart(forecasts):
    """Create forecast chart."""
    try:
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
        if total_forecast.empty and not forecasts.empty:
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
    except:
        return None

def main():
    """Main dashboard function."""
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
        
        # Show file status
        st.markdown("### File Status")
        files_to_check = [
            ("Daily Stats", "data/processed/weather_alerts_daily.csv"),
            ("Anomalies", "data/output/anomaly_results.csv"),
            ("Forecasts", "data/output/forecast_results.csv"),
            ("Processed Alerts", "data/processed/weather_alerts_processed.csv")
        ]
        
        for file_name, file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    st.markdown(f"✓ {file_name}: {size:,} bytes")
                except:
                    st.markdown(f"✓ {file_name}: Exists")
            else:
                st.markdown(f"✗ {file_name}: Missing")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Anomalies", 
        "Forecasts", 
        "Alerts", 
        "Backend Status"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown('<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not data['daily_stats'].empty:
                total_alerts = data['daily_stats']['total_alerts'].sum() if 'total_alerts' in data['daily_stats'].columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Alerts</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">{int(total_alerts)}</p>
                    <p>Across all days in dataset</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
                anomaly_count = data['anomalies']['is_anomaly'].sum()
                st.markdown(f"""
                <div class="metric-card {'warning-card' if anomaly_count > 0 else 'success-card'}">
                    <h3>Detected Anomalies</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: {'#DC2626' if anomaly_count > 0 else '#10B981'};">{int(anomaly_count)}</p>
                    <p>Unusual patterns detected</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if not data['daily_stats'].empty and 'total_alerts' in data['daily_stats'].columns:
                avg_alerts = data['daily_stats']['total_alerts'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Average Daily Alerts</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">{avg_alerts:.1f}</p>
                    <p>Mean alerts per day</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts
        st.markdown('<h3 class="sub-header">Alert Trends</h3>', unsafe_allow_html=True)
        timeline_chart = create_alert_timeline(data['daily_stats'], data['anomalies'])
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="sub-header">Alert Type Distribution</h4>', unsafe_allow_html=True)
            type_chart = create_alert_type_chart(data['daily_stats'])
            if type_chart:
                st.plotly_chart(type_chart, use_container_width=True)
        
        with col2:
            st.markdown('<h4 class="sub-header">Forecast</h4>', unsafe_allow_html=True)
            forecast_chart = create_forecast_chart(data['forecasts'])
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
    
    # Tab 2: Anomalies
    with tab2:
        st.markdown('<h2 class="sub-header">Detected Anomalies</h2>', unsafe_allow_html=True)
        
        if not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomalies = data['anomalies'][data['anomalies']['is_anomaly'] == True]
            
            if not anomalies.empty:
                st.markdown(f"### Found {len(anomalies)} anomalies")
                
                for idx, (date, row) in enumerate(anomalies.iterrows()):
                    with st.container():
                        st.markdown(f"""
                        <div class="insight-card warning-card">
                            <h4>Anomaly #{idx+1} - {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}</h4>
                            <p><strong>Total Alerts:</strong> {row.get('total_alerts', 'N/A')}</p>
                            <p><strong>Anomaly Score:</strong> {row.get('anomaly_score', 'N/A')}</p>
                            <p><strong>Severity:</strong> {row.get('anomaly_severity', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-card success-card">
                    <h4>No Anomalies Detected</h4>
                    <p>No unusual patterns detected in the current dataset.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show anomalies table
            st.markdown("### All Data with Anomaly Flags")
            display_cols = ['total_alerts', 'is_anomaly', 'anomaly_score', 'anomaly_severity']
            available_cols = [col for col in display_cols if col in data['anomalies'].columns]
            
            if available_cols:
                st.dataframe(data['anomalies'][available_cols].tail(30), use_container_width=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <h4>No Anomaly Data Available</h4>
                <p>Run the anomaly detection pipeline to generate anomaly data.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Forecasts
    with tab3:
        st.markdown('<h2 class="sub-header">Weather Alert Forecasts</h2>', unsafe_allow_html=True)
        
        if not data['forecasts'].empty:
            st.markdown("### 7-Day Forecast")
            st.dataframe(data['forecasts'], use_container_width=True)
            
            # Show forecast insights
            st.markdown("### Forecast Insights")
            if not data['forecasts'].empty:
                latest_forecast = data['forecasts'].iloc[-1] if len(data['forecasts']) > 0 else None
                if latest_forecast is not None:
                    st.markdown(f"""
                    <div class="insight-card">
                        <p><strong>Latest Forecast Date:</strong> {latest_forecast['date'].strftime('%Y-%m-%d') if hasattr(latest_forecast['date'], 'strftime') else latest_forecast['date']}</p>
                        <p><strong>Predicted Alerts:</strong> {latest_forecast.get('forecast', 'N/A')}</p>
                        <p><strong>Confidence Range:</strong> {latest_forecast.get('lower_bound', 'N/A')} to {latest_forecast.get('upper_bound', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <h4>No Forecast Data Available</h4>
                <p>Run the forecasting pipeline to generate predictions.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Alerts
    with tab4:
        st.markdown('<h2 class="sub-header">Detailed Alert Data</h2>', unsafe_allow_html=True)
        
        if not data['alerts'].empty:
            st.markdown(f"### Processed Alerts ({len(data['alerts'])} records)")
            
            # Show data summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(data['alerts']))
            with col2:
                if 'severity' in data['alerts'].columns:
                    unique_severities = data['alerts']['severity'].nunique()
                    st.metric("Unique Severity Levels", unique_severities)
            
            # Show sample of data
            st.markdown("### Sample Data")
            st.dataframe(data['alerts'].head(20), use_container_width=True)
            
            # Data statistics
            st.markdown("### Data Statistics")
            if not data['alerts'].empty:
                numeric_cols = data['alerts'].select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(data['alerts'][numeric_cols].describe(), use_container_width=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <h4>No Alert Data Available</h4>
                <p>Run the data collection and preprocessing pipelines to load alert data.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 5: Backend Status
    with tab5:
        st.markdown('<h2 class="sub-header">Backend System Status</h2>', unsafe_allow_html=True)
        
        # Check backend modules
        st.markdown("### Backend Modules Status")
        
        modules_to_check = [
            ("Web Scraping", "scraping.scrape_weather_alerts"),
            ("Data Preprocessing", "preprocessing.preprocess_text"),
            ("Anomaly Detection", "ml.anomaly_detection"),
            ("Forecast Models", "ml.forecast_model")
        ]
        
        for module_name, module_path in modules_to_check:
            try:
                importlib.import_module(module_path)
                st.markdown(f"**{module_name}**: Available")
            except Exception as e:
                st.markdown(f"**{module_name}**: Not Available ({str(e)})")
        
        # Check data files
        st.markdown("### Data File Status")
        
        data_files = [
            ("Raw Alerts", "data/raw/weather_alerts_raw.csv"),
            ("Processed Alerts", "data/processed/weather_alerts_processed.csv"),
            ("Daily Stats", "data/processed/weather_alerts_daily.csv"),
            ("Anomaly Results", "data/output/anomaly_results.csv"),
            ("Forecast Results", "data/output/forecast_results.csv")
        ]
        
        for file_name, file_path in data_files:
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    df = pd.read_csv(file_path)
                    st.markdown(f"**{file_name}**: {size:,} bytes, {len(df)} rows")
                except:
                    st.markdown(f"**{file_name}**: Exists but cannot read")
            else:
                st.markdown(f"**{file_name}**: Missing")
        
        # System info
        st.markdown("### System Information")
        st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"**Python Version**: {sys.version}")
        st.markdown(f"**Pandas Version**: {pd.__version__}")
        st.markdown(f"**Streamlit Version**: {st.__version__}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Weather Anomaly Detection Dashboard v1.0 | Connected Backend System</p>
        <p>Data Source: {data_source} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>National Weather Service Integration | Production-Ready Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Create necessary directories
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/output",
        "models",
        "logs",
        "src"
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create empty files if they don't exist
    required_files = [
        "data/raw/weather_alerts_raw.csv",
        "data/processed/weather_alerts_processed.csv",
        "data/processed/weather_alerts_daily.csv",
        "data/output/anomaly_results.csv",
        "data/output/forecast_results.csv"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            # Create parent directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create empty CSV with headers
            if "anomaly" in file_path:
                pd.DataFrame(columns=['date', 'total_alerts', 'is_anomaly', 'anomaly_score']).to_csv(file_path, index=False)
            elif "forecast" in file_path:
                pd.DataFrame(columns=['date', 'target', 'forecast', 'lower_bound', 'upper_bound']).to_csv(file_path, index=False)
            else:
                pd.DataFrame().to_csv(file_path, index=False)
    
    main()
