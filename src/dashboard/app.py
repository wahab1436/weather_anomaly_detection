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
    with st.sidebar:
        st.markdown("## Dashboard Controls")
        
        # Date range selector
        st.markdown("### Date Range")
        if not data['daily_stats'].empty:
            min_date = data['daily_stats'].index.min()
            max_date = data['daily_stats'].index.max()
            
            date_range = st.date_input(
                "Select date range",
                value=(max_date - timedelta(days=30), max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                # Filter data based on selected range
                mask = (data['daily_stats'].index >= pd.Timestamp(start_date)) & \
                       (data['daily_stats'].index <= pd.Timestamp(end_date))
                filtered_daily_stats = data['daily_stats'][mask]
            else:
                filtered_daily_stats = data['daily_stats']
        else:
            filtered_daily_stats = data['daily_stats']
            st.info("No date range data available")
        
        # Alert type filter
        st.markdown("### Alert Types")
        alert_type_cols = [col for col in data['daily_stats'].columns if col in [
            'flood', 'storm', 'wind', 'winter', 'fire', 
            'heat', 'cold', 'coastal', 'air', 'other'
        ]]
        
        if alert_type_cols:
            selected_types = st.multiselect(
                "Filter by alert type",
                options=alert_type_cols,
                default=alert_type_cols[:3] if len(alert_type_cols) >= 3 else alert_type_cols
            )
        else:
            selected_types = []
            st.info("No alert type data available")
        
        # Region filter
        st.markdown("### Regions")
        if 'alerts' in data and not data['alerts'].empty and 'region' in data['alerts'].columns:
            regions = sorted(data['alerts']['region'].dropna().unique())
            selected_regions = st.multiselect(
                "Filter by region",
                options=regions,
                default=regions[:5] if len(regions) >= 5 else regions
            )
        else:
            selected_regions = []
        
        # Anomaly severity filter
        st.markdown("### Anomaly Severity")
        severity_levels = ['low', 'medium', 'high', 'critical']
        selected_severities = st.multiselect(
            "Filter anomaly severity",
            options=severity_levels,
            default=severity_levels
        )
        
        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Last updated timestamp
        st.markdown("---")
        if os.path.exists('data/output/dashboard_data.csv'):
            last_updated = datetime.fromtimestamp(
                os.path.getmtime('data/output/dashboard_data.csv')
            )
            st.markdown(f"**Last Updated:** {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main dashboard content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Anomaly Analysis", 
        "Forecasts", 
        "Alert Details"
    ])
    
    with tab1:
        # Overview Tab
        st.markdown('<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True)
        
        # Display metrics
        display_metrics(filtered_daily_stats)
        
        # Load insights
        insights = load_insights()
        
        # Insights section
        st.markdown('<h3 class="sub-header">Key Insights</h3>', unsafe_allow_html=True)
        
        for i, insight in enumerate(insights[:5]):  # Show top 5 insights
            st.markdown(f"""
            <div class="insight-card">
                <p style="margin: 0; font-size: 1rem;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert timeline
            fig_timeline = create_alert_timeline(
                filtered_daily_stats, 
                data['anomalies'] if 'anomalies' in data else pd.DataFrame()
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Alert type distribution
            fig_types = create_alert_type_chart(filtered_daily_stats)
            st.plotly_chart(fig_types, use_container_width=True)
        
        # Charts row 2
        col3, col4 = st.columns(2)
        
        with col3:
            # Forecast chart
            fig_forecast = create_forecast_chart(
                data['forecasts'] if 'forecasts' in data else pd.DataFrame()
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col4:
            # Regional map
            fig_map = create_region_map(
                data['alerts'] if 'alerts' in data else pd.DataFrame()
            )
            st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        # Anomaly Analysis Tab
        st.markdown('<h2 class="sub-header">Anomaly Detection Analysis</h2>', unsafe_allow_html=True)
        
        # Anomaly metrics
        if 'anomalies' in data and not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomaly_stats = data['anomalies']['is_anomaly'].value_counts()
            total_anomalies = anomaly_stats.get(True, 0)
            total_days = len(data['anomalies'])
            anomaly_rate = (total_anomalies / total_days * 100) if total_days > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Anomalies", f"{total_anomalies}")
            
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            with col3:
                # Filter by selected severities
                if 'anomaly_severity' in data['anomalies'].columns:
                    filtered_anomalies = data['anomalies'][
                        (data['anomalies']['is_anomaly']) & 
                        (data['anomalies']['anomaly_severity'].isin(selected_severities))
                    ]
                    st.metric("Filtered Anomalies", f"{len(filtered_anomalies)}")
            
            # Anomaly timeline with severity
            if not filtered_daily_stats.empty:
                fig_anomaly_detailed = go.Figure()
                
                # Add total alerts line
                fig_anomaly_detailed.add_trace(
                    go.Scatter(
                        x=filtered_daily_stats.index,
                        y=filtered_daily_stats['total_alerts'],
                        name='Total Alerts',
                        line=dict(color='#3B82F6', width=2),
                        mode='lines'
                    )
                )
                
                # Add anomaly points with severity colors
                if 'anomalies' in data and not data['anomalies'].empty:
                    anomaly_points = data['anomalies'][data['anomalies']['is_anomaly']]
                    
                    if 'anomaly_severity' in anomaly_points.columns:
                        # Color mapping for severities
                        color_map = {
                            'low': '#0EA5E9',
                            'medium': '#F59E0B',
                            'high': '#EF4444',
                            'critical': '#7F1D1D'
                        }
                        
                        for severity in selected_severities:
                            severity_points = anomaly_points[anomaly_points['anomaly_severity'] == severity]
                            if not severity_points.empty:
                                fig_anomaly_detailed.add_trace(
                                    go.Scatter(
                                        x=severity_points.index,
                                        y=severity_points['total_alerts'],
                                        name=f'{severity.capitalize()} Anomaly',
                                        mode='markers',
                                        marker=dict(
                                            color=color_map.get(severity, '#DC2626'),
                                            size=12,
                                            symbol='diamond',
                                            line=dict(width=2, color='white')
                                        )
                                    )
                                )
                
                fig_anomaly_detailed.update_layout(
                    title='Anomaly Detection by Severity',
                    xaxis_title='Date',
                    yaxis_title='Number of Alerts',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_anomaly_detailed, use_container_width=True)
            
            # Anomaly explanations
            st.markdown('<h3 class="sub-header">Anomaly Explanations</h3>', unsafe_allow_html=True)
            
            explanations_file = 'data/output/anomaly_results_explanations.json'
            if os.path.exists(explanations_file):
                with open(explanations_file, 'r') as f:
                    explanations = json.load(f)
                
                for date_str, explanation in list(explanations.items())[:10]:  # Show latest 10
                    severity = explanation.get('severity', 'unknown')
                    css_class = f'anomaly-{severity}' if severity in ['low', 'medium', 'high'] else 'insight-card'
                    
                    st.markdown(f"""
                    <div class="{css_class}">
                        <p style="margin: 0; font-weight: 600;">{date_str} - {severity.upper()} severity</p>
                        <p style="margin: 0.5rem 0 0 0;">Total Alerts: {explanation.get('total_alerts', 'N/A')}</p>
                        <p style="margin: 0.25rem 0 0 0;">Confidence: {explanation.get('confidence', 0):.3f}</p>
                        <p style="margin: 0.5rem 0 0 0; font-weight: 500;">Reasons:</p>
                        <ul style="margin: 0.25rem 0 0 0;">
                            {''.join([f'<li>{reason}</li>' for reason in explanation.get('reasons', [])])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Anomaly table
            display_anomaly_table(data['anomalies'] if 'anomalies' in data else pd.DataFrame())
        
        else:
            st.info("Anomaly detection data not available yet. Run the anomaly detection pipeline first.")
    
    with tab3:
        # Forecasts Tab
        st.markdown('<h2 class="sub-header">Weather Alert Forecasts</h2>', unsafe_allow_html=True)
        
        if 'forecasts' in data and not data['forecasts'].empty:
            # Forecast summary metrics
            total_forecast = data['forecasts'][data['forecasts']['target'] == 'total_alerts']
            
            if not total_forecast.empty:
                avg_forecast = total_forecast['forecast'].mean()
                max_forecast = total_forecast['forecast'].max()
                min_forecast = total_forecast['forecast'].min()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Forecasted Alerts", f"{avg_forecast:.1f}")
                
                with col2:
                    st.metric("Max Forecast", f"{max_forecast:.0f}")
                
                with col3:
                    st.metric("Min Forecast", f"{min_forecast:.0f}")
            
            # Forecast by alert type
            st.markdown('<h3 class="sub-header">Forecast by Alert Type</h3>', unsafe_allow_html=True)
            
            forecast_targets = data['forecasts']['target'].unique()
            
            for target in forecast_targets:
                target_data = data['forecasts'][data['forecasts']['target'] == target]
                
                if not target_data.empty:
                    st.markdown(f"**{target.capitalize()} Alerts Forecast**")
                    
                    fig_target = go.Figure()
                    
                    fig_target.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(target_data['date']),
                            y=target_data['forecast'],
                            name='Forecast',
                            line=dict(color='#3B82F6', width=3),
                            mode='lines'
                        )
                    )
                    
                    # Add confidence interval
                    fig_target.add_trace(
                        go.Scatter(
                            x=pd.concat([pd.to_datetime(target_data['date']), 
                                        pd.to_datetime(target_data['date'])[::-1]]),
                            y=pd.concat([target_data['upper_bound'], 
                                        target_data['lower_bound'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(59, 130, 246, 0.2)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            name='Confidence Interval',
                            showlegend=True
                        )
                    )
                    
                    fig_target.update_layout(
                        xaxis_title='Date',
                        yaxis_title=f'Predicted {target.capitalize()} Alerts',
                        template='plotly_white',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig_target, use_container_width=True)
            
            # Forecast evaluation
            eval_file = 'data/output/forecast_results_evaluation.json'
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
                
                eval_df = pd.DataFrame(eval_results).T.reset_index()
                eval_df.columns = ['Target', 'Mean MAE', 'Std MAE', 'Mean RMSE', 
                                 'Std RMSE', 'Mean MAPE', 'N Folds']
                
                st.dataframe(
                    eval_df,
                    use_container_width=True,
                    hide_index=True
                )
        
        else:
            st.info("Forecast data not available yet. Run the forecasting pipeline first.")
    
    with tab4:
        # Alert Details Tab
        st.markdown('<h2 class="sub-header">Detailed Alert Information</h2>', unsafe_allow_html=True)
        
        # Recent alerts table
        display_recent_alerts(data['alerts'] if 'alerts' in data else pd.DataFrame())
        
        # Alert statistics
        if 'alerts' in data and not data['alerts'].empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Most common alert type
                if 'type' in data['alerts'].columns:
                    common_type = data['alerts']['type'].mode()
                    if not common_type.empty:
                        st.metric("Most Common Alert Type", common_type.iloc[0])
            
            with col2:
                # Most affected region
                if 'region' in data['alerts'].columns:
                    common_region = data['alerts']['region'].mode()
                    if not common_region.empty:
                        st.metric("Most Affected Region", common_region.iloc[0])
            
            with col3:
                # Average alert severity
                if 'severity_score' in data['alerts'].columns:
                    avg_severity = data['alerts']['severity_score'].mean()
                    st.metric("Avg Alert Severity", f"{avg_severity:.2f}")
            
            # Alert text analysis
            st.markdown('<h3 class="sub-header">Alert Text Analysis</h3>', unsafe_allow_html=True)
            
            if 'keywords' in data['alerts'].columns:
                # Extract all keywords
                all_keywords = []
                for kw_list in data['alerts']['keywords'].dropna():
                    if isinstance(kw_list, str):
                        try:
                            kw_list = eval(kw_list)  # Convert string representation of list
                        except:
                            continue
                    all_keywords.extend(kw_list)
                
                if all_keywords:
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    top_keywords = keyword_counts.most_common(20)
                    
                    # Create word cloud or bar chart
                    keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
                    
                    fig_keywords = px.bar(
                        keywords_df,
                        x='Count',
                        y='Keyword',
                        orientation='h',
                        title='Top 20 Keywords in Alert Text'
                    )
                    
                    fig_keywords.update_layout(
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig_keywords, use_container_width=True)
            
            # Alert sentiment analysis
            if 'sentiment_label' in data['alerts'].columns:
                st.markdown('<h3 class="sub-header">Alert Sentiment Distribution</h3>', unsafe_allow_html=True)
                
                sentiment_counts = data['alerts']['sentiment_label'].value_counts()
                
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='Alert Sentiment Distribution'
                )
                
                fig_sentiment.update_layout(
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Weather Anomaly Detection Dashboard v1.0 | Production System</p>
        <p>Data Source: National Weather Service (weather.gov)</p>
        <p>Updates Hourly | Last Scrape: {}
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the dashboard
    main()
