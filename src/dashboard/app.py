"""
Main Streamlit dashboard for Weather Anomaly Detection.
Professional, production-ready interface.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional
import logging

# Configure page
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
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
    }
    .insight-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #DBEAFE;
    }
    .anomaly-high {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
    }
    .anomaly-medium {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
    }
    .anomaly-low {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
    }
    .last-updated {
        font-size: 0.875rem;
        color: #6B7280;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data() -> Dict[str, pd.DataFrame]:
    """Load all data for the dashboard with caching."""
    data_files = {
        'daily_stats': 'data/processed/weather_alerts_daily.csv',
        'anomalies': 'data/output/anomaly_results.csv',
        'forecasts': 'data/output/forecast_results.csv',
        'alerts': 'data/processed/weather_alerts_processed.csv',
        'dashboard': 'data/output/dashboard_data.csv'
    }
    
    data = {}
    
    for name, filepath in data_files.items():
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                
                # Parse dates for time series data
                if name in ['daily_stats', 'anomalies', 'dashboard']:
                    if 'issued_date' in df.columns:
                        df['issued_date'] = pd.to_datetime(df['issued_date'])
                        df = df.set_index('issued_date')
                
                data[name] = df
                logger.info(f"Loaded {len(df)} rows from {filepath}")
            else:
                logger.warning(f"File not found: {filepath}")
                data[name] = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            data[name] = pd.DataFrame()
    
    return data

@st.cache_data(ttl=3600)
def load_insights() -> List[str]:
    """Load or generate insights for the dashboard."""
    insights_file = 'data/output/insights.json'
    
    try:
        if os.path.exists(insights_file):
            with open(insights_file, 'r') as f:
                insights_data = json.load(f)
                return insights_data.get('insights', [])
    except Exception as e:
        logger.error(f"Error loading insights: {str(e)}")
    
    # Fallback insights if file doesn't exist
    return [
        "System is initializing. First data load in progress.",
        "Monitoring weather alerts for anomaly detection.",
        "Forecast models will generate predictions once sufficient data is available."
    ]

def create_alert_timeline(daily_stats: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    """Create timeline chart with alerts and anomalies."""
    if daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add total alerts line
    fig.add_trace(
        go.Scatter(
            x=daily_stats.index,
            y=daily_stats['total_alerts'],
            name='Total Alerts',
            line=dict(color='#3B82F6', width=2),
            mode='lines'
        ),
        secondary_y=False
    )
    
    # Add 7-day moving average
    if '7_day_avg' in daily_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats['7_day_avg'],
                name='7-Day Average',
                line=dict(color='#6B7280', width=1, dash='dash'),
                mode='lines'
            ),
            secondary_y=False
        )
    
    # Add anomalies if available
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly']]
        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points.index,
                    y=anomaly_points['total_alerts'],
                    name='Anomalies',
                    mode='markers',
                    marker=dict(
                        color='#DC2626',
                        size=10,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    )
                ),
                secondary_y=False
            )
    
    # Add severity score if available
    if 'severity_score' in daily_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats['severity_score'] * 100,
                name='Severity Score (%)',
                line=dict(color='#EF4444', width=1),
                mode='lines'
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title='Daily Weather Alerts with Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        yaxis2_title='Severity Score (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_alert_type_chart(daily_stats: pd.DataFrame) -> go.Figure:
    """Create chart showing alert type distribution."""
    # Identify alert type columns
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
    
    # Get recent data (last 30 days)
    recent_data = daily_stats.tail(30)
    
    # Calculate totals
    type_totals = recent_data[alert_type_cols].sum().sort_values(ascending=False)
    
    # Create bar chart
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
        height=400
    )
    
    return fig

def create_forecast_chart(forecasts: pd.DataFrame) -> go.Figure:
    """Create forecast visualization."""
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
    
    # Filter for total alerts forecast
    total_forecast = forecasts[forecasts['target'] == 'total_alerts']
    
    if total_forecast.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No total alerts forecast available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Convert date strings to datetime
    total_forecast['date'] = pd.to_datetime(total_forecast['date'])
    
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(
        go.Scatter(
            x=total_forecast['date'],
            y=total_forecast['forecast'],
            name='Forecast',
            line=dict(color='#3B82F6', width=3),
            mode='lines'
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=pd.concat([total_forecast['date'], total_forecast['date'][::-1]]),
            y=pd.concat([total_forecast['upper_bound'], total_forecast['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval',
            showlegend=True
        )
    )
    
    fig.update_layout(
        title='7-Day Alert Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Alerts',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_region_map(alerts: pd.DataFrame) -> go.Figure:
    """Create geographical visualization of alerts by region."""
    if alerts.empty or 'region' not in alerts.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No regional data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Count alerts by region
    region_counts = alerts['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    
    # Simplified region to state mapping (in production, use proper geocoding)
    state_abbr = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Create choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=list(state_abbr.keys()),
        z=[10] * len(state_abbr),  # Placeholder values
        locationmode='USA-states',
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title_text='Alert Activity by Region',
        geo_scope='usa',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def display_metrics(daily_stats: pd.DataFrame) -> None:
    """Display key metrics in columns."""
    if daily_stats.empty:
        st.warning("No data available for metrics display")
        return
    
    # Calculate metrics
    total_alerts = daily_stats['total_alerts'].sum() if 'total_alerts' in daily_stats.columns else 0
    avg_daily = daily_stats['total_alerts'].mean() if 'total_alerts' in daily_stats.columns else 0
    max_daily = daily_stats['total_alerts'].max() if 'total_alerts' in daily_stats.columns else 0
    
    # Recent metrics
    recent = daily_stats.tail(7)
    recent_avg = recent['total_alerts'].mean() if not recent.empty else 0
    prev_week = daily_stats.iloc[-14:-7]['total_alerts'].mean() if len(daily_stats) >= 14 else 0
    
    # Calculate change
    if prev_week > 0:
        change_pct = ((recent_avg - prev_week) / prev_week) * 100
    else:
        change_pct = 0
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.1rem; color: #6B7280;">Total Alerts</h3>
            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700; color: #1E3A8A;">{int(total_alerts):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.1rem; color: #6B7280;">Avg Daily Alerts</h3>
            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700; color: #1E3A8A;">{avg_daily:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.1rem; color: #6B7280;">Max Daily</h3>
            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700; color: #1E3A8A;">{int(max_daily)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        change_color = "#10B981" if change_pct <= 0 else "#EF4444"
        change_symbol = "▼" if change_pct <= 0 else "▲"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.1rem; color: #6B7280;">7-Day Trend</h3>
            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700; color: {change_color};">
                {change_symbol} {abs(change_pct):.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_anomaly_table(anomalies: pd.DataFrame) -> None:
    """Display table of detected anomalies."""
    if anomalies.empty or 'is_anomaly' not in anomalies.columns:
        st.info("No anomalies detected")
        return
    
    anomaly_data = anomalies[anomalies['is_anomaly']].copy()
    
    if anomaly_data.empty:
        st.info("No anomalies detected in the current dataset")
        return
    
    # Prepare table data
    anomaly_data = anomaly_data.reset_index()
    
    # Select and rename columns
    display_cols = ['issued_date', 'total_alerts']
    
    if 'anomaly_severity' in anomaly_data.columns:
        display_cols.append('anomaly_severity')
    if 'anomaly_confidence' in anomaly_data.columns:
        display_cols.append('anomaly_confidence')
    
    anomaly_display = anomaly_data[display_cols].copy()
    
    # Format date
    anomaly_display['issued_date'] = anomaly_display['issued_date'].dt.strftime('%Y-%m-%d')
    
    # Format confidence
    if 'anomaly_confidence' in anomaly_display.columns:
        anomaly_display['anomaly_confidence'] = anomaly_display['anomaly_confidence'].apply(
            lambda x: f"{x:.3f}"
        )
    
    # Rename columns
    column_names = {
        'issued_date': 'Date',
        'total_alerts': 'Alerts',
        'anomaly_severity': 'Severity',
        'anomaly_confidence': 'Confidence'
    }
    
    anomaly_display = anomaly_display.rename(columns=column_names)
    
    # Display table
    st.subheader("Detected Anomalies")
    st.dataframe(
        anomaly_display,
        use_container_width=True,
        hide_index=True
    )

def display_recent_alerts(alerts: pd.DataFrame) -> None:
    """Display table of recent alerts."""
    if alerts.empty:
        st.info("No alert data available")
        return
    
    # Get recent alerts
    if 'issued_date' in alerts.columns:
        alerts_sorted = alerts.sort_values('issued_date', ascending=False)
        recent_alerts = alerts_sorted.head(20)
    else:
        recent_alerts = alerts.head(20)
    
    # Prepare display
    display_cols = []
    if 'issued_date' in recent_alerts.columns:
        display_cols.append('issued_date')
    if 'type' in recent_alerts.columns:
        display_cols.append('type')
    if 'region' in recent_alerts.columns:
        display_cols.append('region')
    if 'title' in recent_alerts.columns:
        display_cols.append('title')
    
    if not display_cols:
        st.info("No alert details available")
        return
    
    recent_display = recent_alerts[display_cols].copy()
    
    # Format date
    if 'issued_date' in recent_display.columns:
        recent_display['issued_date'] = pd.to_datetime(recent_display['issued_date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Rename columns
    column_names = {
        'issued_date': 'Time',
        'type': 'Type',
        'region': 'Region',
        'title': 'Alert'
    }
    
    recent_display = recent_display.rename(columns=column_names)
    
    # Display table
    st.subheader("Recent Alerts")
    st.dataframe(
        recent_display,
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">Weather Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.s
