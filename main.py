"""
Main Weather Anomaly Detection Dashboard - Streamlit Entry Point
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load or generate dashboard data."""
    try:
        # Try to load processed data
        if os.path.exists("data/processed/weather_alerts_daily.csv"):
            df = pd.read_csv("data/processed/weather_alerts_daily.csv")
            if 'issued_date' in df.columns:
                df['date'] = pd.to_datetime(df['issued_date'])
            return df, "Live Data"
    except Exception as e:
        st.warning(f"Could not load data: {e}")
    
    # Generate sample data for demo
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'total_alerts': np.random.randint(10, 50, 30),
        'flood': np.random.randint(0, 15, 30),
        'storm': np.random.randint(0, 20, 30),
        'wind': np.random.randint(0, 10, 30),
        'winter': np.random.randint(0, 8, 30),
        'fire': np.random.randint(0, 5, 30),
        'heat': np.random.randint(0, 3, 30),
        'cold': np.random.randint(0, 4, 30)
    })
    
    return sample_data, "Sample Data (Demo Mode)"

def create_alert_timeline(df):
    """Create timeline chart."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_alerts'],
        mode='lines',
        name='Total Alerts',
        line=dict(color='#3B82F6', width=2)
    ))
    
    fig.update_layout(
        title='Daily Weather Alerts',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_alert_type_chart(df):
    """Create alert type distribution chart."""
    import plotly.graph_objects as go
    
    alert_types = ['flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold']
    type_totals = df[alert_types].sum()
    
    fig = go.Figure(data=[
        go.Bar(
            x=type_totals.index,
            y=type_totals.values,
            marker_color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4'],
            text=type_totals.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Alert Type Distribution',
        xaxis_title='Alert Type',
        yaxis_title='Number of Alerts',
        template='plotly_white',
        height=300
    )
    
    return fig

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">Weather Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    st.write("### Professional Weather Alert Monitoring System")
    
    # Load data
    df, data_source = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Dashboard Controls")
        st.markdown(f"**Data Source:** {data_source}")
        
        # Date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=7), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Alert type filter
        st.markdown("### Filter by Alert Type")
        alert_types = ['flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold']
        selected_types = st.multiselect(
            "Select alert types",
            options=alert_types,
            default=alert_types[:3]
        )
        
        # System controls
        st.markdown("---")
        st.markdown("### System Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Refresh Dashboard"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("Run Data Collection"):
                try:
                    # Try to run scraping
                    from scraping.scrape_weather_alerts import main as scrape_main
                    with st.spinner("Collecting data..."):
                        scrape_main()
                    st.success("Data collection complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # System info
        st.markdown("---")
        st.markdown("### System Status")
        
        if data_source == "Live Data":
            st.success("✓ Live data system active")
        else:
            st.info("⚠ Running in demo mode")
        
        if os.path.exists("data/processed/weather_alerts_daily.csv"):
            last_updated = datetime.fromtimestamp(
                os.path.getmtime("data/processed/weather_alerts_daily.csv")
            )
            st.markdown(f"**Last Updated:** {last_updated.strftime('%Y-%m-%d %H:%M')}")
    
    # Metrics
    st.markdown('<h2 class="sub-header">Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = df['total_alerts'].sum()
        st.metric("Total Alerts", f"{int(total_alerts):,}")
    
    with col2:
        avg_daily = df['total_alerts'].mean()
        st.metric("Avg Daily", f"{avg_daily:.1f}")
    
    with col3:
        max_daily = df['total_alerts'].max()
        st.metric("Max Daily", f"{int(max_daily)}")
    
    with col4:
        recent_avg = df.tail(7)['total_alerts'].mean()
        st.metric("7-Day Avg", f"{recent_avg:.1f}")
    
    # Charts
    st.markdown('<h2 class="sub-header">Alert Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_timeline = create_alert_timeline(df)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        fig_types = create_alert_type_chart(df)
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Alert type breakdown
    st.markdown('<h2 class="sub-header">Alert Type Breakdown</h2>', unsafe_allow_html=True)
    
    if selected_types:
        selected_df = df[['date'] + selected_types].tail(10)
        st.dataframe(selected_df, use_container_width=True)
    
    # Insights
    st.markdown('<h2 class="sub-header">System Insights</h2>', unsafe_allow_html=True)
    
    insights = [
        "System is monitoring weather alerts for anomaly detection.",
        "Data collection runs hourly from weather.gov.",
        "Anomaly detection models identify unusual alert patterns.",
        "Forecast models predict future alert trends."
    ]
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            <p style="margin: 0;">{insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System information
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Weather Anomaly Detection Dashboard v1.0 | Professional Production System</p>
        <p>Data Source: {data_source} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>National Weather Service Integration | Hourly Updates</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    main()
