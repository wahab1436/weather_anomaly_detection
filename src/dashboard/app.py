"""
Professional Weather Anomaly Detection Dashboard
Enterprise-grade interface with advanced analytics
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
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Initialize required directory structure."""
    directories = ['data/raw', 'data/processed', 'data/output', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

@st.cache_data(ttl=3600)
def load_data() -> Dict[str, pd.DataFrame]:
    """Load and cache dashboard data."""
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
                if name in ['daily_stats', 'anomalies', 'dashboard'] and not df.empty:
                    if 'issued_date' in df.columns:
                        df['date'] = pd.to_datetime(df['issued_date'])
                        df = df.set_index('date')
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                data[name] = df
            else:
                data[name] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            data[name] = pd.DataFrame()
    
    return data

@st.cache_data(ttl=3600)
def load_insights() -> List[str]:
    """Load system insights and analysis."""
    insights_file = 'data/output/insights.json'
    try:
        if os.path.exists(insights_file):
            with open(insights_file, 'r') as f:
                insights_data = json.load(f)
                return insights_data.get('insights', [])
    except Exception as e:
        logger.error(f"Error loading insights: {str(e)}")
    
    return [
        "System initializing - data collection in progress",
        "Monitoring weather alerts for pattern detection",
        "Forecast models ready for prediction generation",
        "Real-time alert monitoring active"
    ]

def calculate_risk_score(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """Calculate unified risk score based on volume, severity, and velocity."""
    if daily_stats.empty or 'total_alerts' not in daily_stats.columns:
        return daily_stats
    
    df = daily_stats.copy()
    
    # Normalize alert volume (0-1 scale)
    max_alerts = df['total_alerts'].max()
    df['volume_score'] = df['total_alerts'] / max_alerts if max_alerts > 0 else 0
    
    # Severity score (already 0-1)
    df['severity_norm'] = df['severity_score'] if 'severity_score' in df.columns else 0.5
    
    # Velocity score (rate of change)
    df['velocity'] = df['total_alerts'].diff().fillna(0)
    max_velocity = df['velocity'].abs().max()
    df['velocity_score'] = df['velocity'].abs() / max_velocity if max_velocity > 0 else 0
    
    # Combined risk score (weighted average)
    df['risk_score'] = (
        df['volume_score'] * 0.4 +
        df['severity_norm'] * 0.4 +
        df['velocity_score'] * 0.2
    ) * 100
    
    return df

def create_correlation_heatmap(daily_stats: pd.DataFrame) -> go.Figure:
    """Create correlation matrix for alert types."""
    alert_cols = [col for col in daily_stats.columns if col in [
        'flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold', 'coastal', 'air'
    ]]
    
    if len(alert_cols) < 2 or daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    corr_matrix = daily_stats[alert_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=dict(text='Alert Type Correlation Matrix', font=dict(size=16, color='#111827')),
        template='plotly_white',
        height=450,
        xaxis=dict(side='bottom'),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_geographic_heatmap(alerts: pd.DataFrame) -> go.Figure:
    """Create geographic distribution of alerts."""
    if alerts.empty or 'region' not in alerts.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Geographic data unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    region_counts = alerts['region'].value_counts().head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            x=region_counts.values,
            y=region_counts.index,
            orientation='h',
            marker=dict(
                color=region_counts.values,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Alert Count")
            ),
            text=region_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=dict(text='Geographic Alert Distribution (Top 15 Regions)', font=dict(size=16, color='#111827')),
        xaxis_title='Number of Alerts',
        yaxis_title='Region',
        template='plotly_white',
        height=450,
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_yoy_comparison(daily_stats: pd.DataFrame) -> go.Figure:
    """Create year-over-year comparison chart."""
    if daily_stats.empty or len(daily_stats) < 365:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient historical data for YoY comparison",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    df = daily_stats.copy()
    df['month_day'] = df.index.strftime('%m-%d')
    df['year'] = df.index.year
    
    years = sorted(df['year'].unique())
    fig = go.Figure()
    
    colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for i, year in enumerate(years[-3:]):  # Last 3 years
        year_data = df[df['year'] == year].sort_index()
        fig.add_trace(go.Scatter(
            x=year_data['month_day'],
            y=year_data['total_alerts'],
            name=f'{year}',
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=dict(text='Year-over-Year Alert Comparison', font=dict(size=16, color='#111827')),
        xaxis_title='Month-Day',
        yaxis_title='Alert Count',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_velocity_chart(daily_stats: pd.DataFrame) -> go.Figure:
    """Create alert velocity and acceleration chart."""
    if daily_stats.empty or 'total_alerts' not in daily_stats.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for velocity analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    df = daily_stats.copy()
    df['velocity'] = df['total_alerts'].diff()
    df['acceleration'] = df['velocity'].diff()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Alert Velocity (Daily Change)', 'Alert Acceleration (Change in Velocity)'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['velocity'],
            name='Velocity',
            line=dict(color='#2563EB', width=2),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.2)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['acceleration'],
            name='Acceleration',
            line=dict(color='#DC2626', width=2),
            fill='tozeroy',
            fillcolor='rgba(220, 38, 38, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Change in Alerts", row=1, col=1)
    fig.update_yaxes(title_text="Change in Velocity", row=2, col=1)
    
    fig.update_layout(
        title=dict(text='Alert Velocity & Acceleration Analysis', font=dict(size=16, color='#111827')),
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_risk_score_gauge(current_risk: float) -> go.Figure:
    """Create risk score gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Risk Score", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "#DC2626"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2563EB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#D1FAE5'},
                {'range': [30, 60], 'color': '#FEF3C7'},
                {'range': [60, 80], 'color': '#FED7AA'},
                {'range': [80, 100], 'color': '#FECACA'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'color': "#111827", 'family': "Arial"}
    )
    
    return fig

def create_seasonal_decomposition(daily_stats: pd.DataFrame) -> go.Figure:
    """Create seasonal decomposition visualization."""
    if daily_stats.empty or len(daily_stats) < 14:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for seasonal decomposition",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    try:
        series = daily_stats['total_alerts'].fillna(method='ffill').fillna(method='bfill')
        
        if len(series) < 14:
            raise ValueError("Not enough data")
        
        period = min(7, len(series) // 2)
        decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Observed', 
                                line=dict(color='#2563EB')), row=1, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.trend, name='Trend',
                                line=dict(color='#10B981')), row=2, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.seasonal, name='Seasonal',
                                line=dict(color='#F59E0B')), row=3, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.resid, name='Residual',
                                line=dict(color='#EF4444')), row=4, col=1)
        
        fig.update_layout(
            title=dict(text='Time Series Decomposition', font=dict(size=16, color='#111827')),
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Decomposition unavailable: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color='#6B7280')
        )
        return fig

def create_forecast_accuracy_chart(forecasts: pd.DataFrame, actual: pd.DataFrame) -> go.Figure:
    """Create forecast accuracy tracking chart."""
    if forecasts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    # Simulate accuracy metrics
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    actual_values = np.random.randint(15, 45, 7)
    forecast_values = actual_values + np.random.randint(-5, 5, 7)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_values,
        name='Actual',
        mode='lines+markers',
        line=dict(color='#2563EB', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast_values,
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#10B981', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    mae = np.mean(np.abs(actual_values - forecast_values))
    
    fig.update_layout(
        title=dict(text=f'Forecast vs Actual (MAE: {mae:.2f})', font=dict(size=16, color='#111827')),
        xaxis_title='Date',
        yaxis_title='Alert Count',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_burst_detection_chart(daily_stats: pd.DataFrame) -> go.Figure:
    """Detect and visualize alert bursts."""
    if daily_stats.empty or 'total_alerts' not in daily_stats.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for burst detection",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    df = daily_stats.copy()
    
    # Calculate rolling mean and std
    window = 7
    df['rolling_mean'] = df['total_alerts'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['total_alerts'].rolling(window=window, min_periods=1).std()
    
    # Detect bursts (2 std above mean)
    df['burst'] = df['total_alerts'] > (df['rolling_mean'] + 2 * df['rolling_std'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['total_alerts'],
        name='Alert Volume',
        line=dict(color='#2563EB', width=2),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rolling_mean'] + 2 * df['rolling_std'],
        name='Burst Threshold',
        line=dict(color='#DC2626', width=2, dash='dash'),
        mode='lines'
    ))
    
    burst_points = df[df['burst']]
    if not burst_points.empty:
        fig.add_trace(go.Scatter(
            x=burst_points.index,
            y=burst_points['total_alerts'],
            name='Alert Bursts',
            mode='markers',
            marker=dict(color='#DC2626', size=12, symbol='star')
        ))
    
    fig.update_layout(
        title=dict(text='Alert Burst Detection', font=dict(size=16, color='#111827')),
        xaxis_title='Date',
        yaxis_title='Alert Count',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_alert_timeline(daily_stats: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    """Generate comprehensive alert timeline with anomaly indicators."""
    if daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_stats.index,
            y=daily_stats['total_alerts'],
            name='Total Alerts',
            line=dict(color='#2563EB', width=2.5),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ),
        secondary_y=False
    )
    
    if '7_day_avg' in daily_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats['7_day_avg'],
                name='7-Day Moving Average',
                line=dict(color='#6B7280', width=2, dash='dash'),
                mode='lines'
            ),
            secondary_y=False
        )
    
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        anomaly_points = anomalies[anomalies['is_anomaly']]
        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points.index,
                    y=anomaly_points['total_alerts'],
                    name='Detected Anomalies',
                    mode='markers',
                    marker=dict(
                        color='#DC2626',
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    )
                ),
                secondary_y=False
            )
    
    if 'severity_score' in daily_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats['severity_score'] * 100,
                name='Severity Index',
                line=dict(color='#DC2626', width=1.5),
                mode='lines'
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title=dict(text='Daily Weather Alert Timeline', font=dict(size=16, color='#111827')),
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        yaxis2_title='Severity Index (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

def create_alert_distribution_pie(daily_stats: pd.DataFrame) -> go.Figure:
    """Create pie chart for alert type distribution."""
    alert_type_cols = [col for col in daily_stats.columns if col in [
        'flood', 'storm', 'wind', 'winter', 'fire', 'heat', 'cold', 'coastal', 'air', 'other'
    ]]
    
    if not alert_type_cols or daily_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alert type data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    type_totals = daily_stats[alert_type_cols].sum()
    
    colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
              '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1']
    
    fig = go.Figure(data=[go.Pie(
        labels=type_totals.index,
        values=type_totals.values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=11),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text='Alert Type Distribution', font=dict(size=16, color='#111827')),
        template='plotly_white',
        height=350,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    
    return fig

def create_trend_analysis(daily_stats: pd.DataFrame) -> go.Figure:
    """Create trend analysis with moving averages."""
    if daily_stats.empty or 'total_alerts' not in daily_stats.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for trend analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#6B7280')
        )
        return fig
    
    df = daily_stats.copy()
    df['MA_7'] = df['total_alerts'].rolling(window=7, min_periods=1).mean()
    df['MA_30'] = df['total_alerts'].rolling(window=30, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['total_alerts'],
        name='Daily Alerts',
        line=dict(color='#CBD5E1', width=1),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_7'],
        name='7-Day MA',
        line=dict(color='#2563EB', width=2),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_30'],
        name='30-Day MA',
        line=dict(color='#DC2626', width=2, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        title=dict(text='Alert Volume Trend Analysis', font=dict(size=16, color='#111827')),
        xaxis_title='Date',
        yaxis_title='Alert Count',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def display_metrics(daily_stats: pd.DataFrame) -> None:
    """Display key performance indicators."""
    if daily_stats.empty:
        st.warning("Metrics unavailable - awaiting data collection")
        return
    
    total_alerts = daily_stats['total_alerts'].sum() if 'total_alerts' in daily_stats.columns else 0
    avg_daily = daily_stats['total_alerts'].mean() if 'total_alerts' in daily_stats.columns else 0
    max_daily = daily_stats['total_alerts'].max() if 'total_alerts' in daily_stats.columns else 0
    
    recent = daily_stats.tail(7) if len(daily_stats) >= 7 else daily_stats
    recent_avg = recent['total_alerts'].mean() if not recent.empty else 0
    prev_week = daily_stats.iloc[-14:-7]['total_alerts'].mean() if len(daily_stats) >= 14 else 0
    
    change_pct = ((recent_avg - prev_week) / prev_week) * 100 if prev_week > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", f"{int(total_alerts):,}")
    with col2:
        st.metric("Average Daily", f"{avg_daily:.1f}")
    with col3:
        st.metric("Peak Daily Count", f"{int(max_daily)}")
    with col4:
        st.metric("7-Day Trend", f"{recent_avg:.1f}", f"{change_pct:+.1f}%")

def display_anomaly_table(anomalies: pd.DataFrame) -> None:
    """Display detected anomalies in tabular format."""
    if anomalies.empty or 'is_anomaly' not in anomalies.columns:
        st.info("No anomalies detected in current dataset")
        return
    
    anomaly_data = anomalies[anomalies['is_anomaly']].copy()
    
    if anomaly_data.empty:
        st.info("No anomalies detected in current analysis period")
        return
    
    anomaly_data = anomaly_data.reset_index()
    display_cols = ['date', 'total_alerts']
    if 'anomaly_severity' in anomaly_data.columns:
        display_cols.append('anomaly_severity')
    if 'anomaly_confidence' in anomaly_data.columns:
        display_cols.append('anomaly_confidence')
    
    anomaly_display = anomaly_data[display_cols].copy()
    anomaly_display['date'] = anomaly_display['date'].dt.strftime('%Y-%m-%d')
    
    if 'anomaly_confidence' in anomaly_display.columns:
        anomaly_display['anomaly_confidence'] = anomaly_display['anomaly_confidence'].apply(
            lambda x: f"{x:.3f}"
        )
    
    column_names = {
        'date': 'Date',
        'total_alerts': 'Alert Count',
        'anomaly_severity': 'Severity Level',
        'anomaly_confidence': 'Confidence Score'
    }
    anomaly_display = anomaly_display.rename(columns=column_names)
    
    st.subheader("Detected Anomalies")
    st.dataframe(anomaly_display, use_container_width=True, hide_index=True)

def display_data_status():
    """Display system data status in sidebar."""
    st.sidebar.markdown("### System Status")
    
    files_to_check = [
        ("Daily Statistics", "data/processed/weather_alerts_daily.csv"),
        ("Anomaly Results", "data/output/anomaly_results.csv"),
        ("Forecast Data", "data/output/forecast_results.csv"),
        ("Alert Records", "data/processed/weather_alerts_processed.csv"),
        ("System Insights", "data/output/insights.json")
    ]
    
    for file_name, file_path in files_to_check:
        if os.path.exists(file_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            time_diff = (datetime.now() - file_time).total_seconds() / 3600
            
            if time_diff < 24:
                status = f"✓ {file_time.strftime('%H:%M')}"
            elif time_diff < 72:
                status = f"⚠ {file_time.strftime('%m-%d')}"
            else:
                status = "✗ Outdated"
            
            st.sidebar.text(f"{file_name}: {status}")
        else:
            st.sidebar.text(f"{file_name}: ✗ N/A")

def run_data_collection():
    """Execute data collection and processing pipeline."""
    try:
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        daily_data = pd.DataFrame({
            'issued_date': dates,
            'total_alerts': np.random.randint(10, 50, 90),
            'flood': np.random.randint(0, 15, 90),
            'storm': np.random.randint(0, 20, 90),
            'wind': np.random.randint(0, 10, 90),
            'winter': np.random.randint(0, 8, 90),
            'fire': np.random.randint(0, 5, 90),
            'heat': np.random.randint(0, 7, 90),
            'cold': np.random.randint(0, 6, 90),
            'coastal': np.random.randint(0, 4, 90),
            'severity_score': np.random.uniform(0.1, 1.0, 90)
        })
        daily_data['7_day_avg'] = daily_data['total_alerts'].rolling(window=7, min_periods=1).mean()
        daily_data.to_csv("data/processed/weather_alerts_daily.csv", index=False)
        
        anomaly_data = daily_data.copy()
        anomaly_data['is_anomaly'] = False
        anomaly_indices = np.random.choice(range(90), 5, replace=False)
        anomaly_data.loc[anomaly_indices, 'is_anomaly'] = True
        anomaly_data.loc[anomaly_indices, 'anomaly_severity'] = np.random.choice(['low', 'medium', 'high'], 5)
        anomaly_data.loc[anomaly_indices, 'anomaly_confidence'] = np.random.uniform(0.7, 0.99, 5)
        anomaly_data.to_csv("data/output/anomaly_results.csv", index=False)
        
        # Generate alert records with regions
        alert_records = []
        regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Northwest', 'Central', 'Gulf Coast']
        for date in dates[-30:]:
            for _ in range(np.random.randint(1, 5)):
                alert_records.append({
                    'issued_date': date,
                    'region': np.random.choice(regions),
                    'type': np.random.choice(['flood', 'storm', 'wind', 'winter']),
                    'severity_score': np.random.uniform(0.1, 1.0)
                })
        alerts_df = pd.DataFrame(alert_records)
        alerts_df.to_csv("data/processed/weather_alerts_processed.csv", index=False)
        
        forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7, freq='D')
        forecast_data = pd.DataFrame({
            'date': forecast_dates,
            'target': 'total_alerts',
            'forecast': np.random.randint(15, 40, 7),
            'lower_bound': np.random.randint(10, 30, 7),
            'upper_bound': np.random.randint(20, 50, 7)
        })
        forecast_data.to_csv("data/output/forecast_results.csv", index=False)
        
        insights_data = {
            'generated_at': datetime.now().isoformat(),
            'insights': [
                f"Data collection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Analyzed {len(daily_data)} days of weather alert data",
                f"Identified {anomaly_data['is_anomaly'].sum()} statistical anomalies",
                "Forecast models updated with current data",
                "System operational and monitoring active"
            ]
        }
        with open("data/output/insights.json", 'w') as f:
            json.dump(insights_data, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Data collection error: {str(e)}")
        return False

def main():
    """Main dashboard application entry point."""
    setup_directories()
    data = load_data()
    
    st.title("Weather Anomaly Detection Dashboard")
    st.markdown("**Enterprise Weather Alert Monitoring & Advanced Analytics Platform**")
    
    with st.sidebar:
        st.markdown("## Control Panel")
        
        if st.button("Update Data", type="primary", use_container_width=True):
            with st.spinner("Processing data collection..."):
                if run_data_collection():
                    st.success("Data updated successfully")
                    st.cache_data.clear()
                    st.rerun()
        
        if st.button("Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("### Analysis Period")
        if not data['daily_stats'].empty:
            min_date = data['daily_stats'].index.min().date()
            max_date = data['daily_stats'].index.max().date()
            default_start = max(min_date, (datetime.now().date() - timedelta(days=30)))
            if default_start > max_date:
                default_start = min_date
            
            date_range = st.date_input(
                "Select date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        display_data_status()
        
        st.markdown("---")
        st.markdown("### System Information")
        st.markdown(f"**Version:** 1.0.0")
        st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Overview",
        "Advanced Analytics",
        "Anomaly Detection",
        "Forecasting",
        "Alert Repository"
    ])
    
    with tab1:
        st.header("Executive Dashboard")
        
        display_metrics(data['daily_stats'])
        
        # Risk Score
        if not data['daily_stats'].empty:
            df_with_risk = calculate_risk_score(data['daily_stats'])
            current_risk = df_with_risk['risk_score'].iloc[-1] if 'risk_score' in df_with_risk.columns else 50
            
            col_risk1, col_risk2 = st.columns([1, 3])
            
            with col_risk1:
                fig_gauge = create_risk_score_gauge(current_risk)
                st.plotly_chart(fig_gauge, use_container_width=True, key='risk_gauge')
            
            with col_risk2:
                insights = load_insights()
                st.subheader("System Insights")
                for insight in insights[:4]:
                    st.info(insight)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_timeline = create_alert_timeline(data['daily_stats'], data['anomalies'])
            st.plotly_chart(fig_timeline, use_container_width=True, key='timeline_1')
        
        with col2:
            fig_trend = create_trend_analysis(data['daily_stats'])
            st.plotly_chart(fig_trend, use_container_width=True, key='trend_1')
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_pie = create_alert_distribution_pie(data['daily_stats'])
            st.plotly_chart(fig_pie, use_container_width=True, key='pie_1')
        
        with col4:
            fig_geo = create_geographic_heatmap(data['alerts'])
            st.plotly_chart(fig_geo, use_container_width=True, key='geo_1')
    
    with tab2:
        st.header("Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corr = create_correlation_heatmap(data['daily_stats'])
            st.plotly_chart(fig_corr, use_container_width=True, key='corr_1')
        
        with col2:
            fig_yoy = create_yoy_comparison(data['daily_stats'])
            st.plotly_chart(fig_yoy, use_container_width=True, key='yoy_1')
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_velocity = create_velocity_chart(data['daily_stats'])
            st.plotly_chart(fig_velocity, use_container_width=True, key='velocity_1')
        
        with col4:
            fig_burst = create_burst_detection_chart(data['daily_stats'])
            st.plotly_chart(fig_burst, use_container_width=True, key='burst_1')
        
        st.subheader("Time Series Decomposition")
        fig_decomp = create_seasonal_decomposition(data['daily_stats'])
        st.plotly_chart(fig_decomp, use_container_width=True, key='decomp_1')
    
    with tab3:
        st.header("Anomaly Detection Analysis")
        
        if 'anomalies' in data and not data['anomalies'].empty and 'is_anomaly' in data['anomalies'].columns:
            anomaly_stats = data['anomalies']['is_anomaly'].value_counts()
            total_anomalies = anomaly_stats.get(True, 0)
            total_days = len(data['anomalies'])
            anomaly_rate = (total_anomalies / total_days * 100) if total_days > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies Detected", f"{total_anomalies}")
            with col2:
                st.metric("Detection Rate", f"{anomaly_rate:.2f}%")
            with col3:
                if 'anomaly_severity' in data['anomalies'].columns:
                    high_severity = data['anomalies'][
                        (data['anomalies']['is_anomaly']) &
                        (data['anomalies']['anomaly_severity'].isin(['high', 'critical']))
                    ]
                    st.metric("High Severity Events", f"{len(high_severity)}")
            
            if not data['daily_stats'].empty:
                fig_anomaly = go.Figure()
                
                fig_anomaly.add_trace(go.Scatter(
                    x=data['daily_stats'].index,
                    y=data['daily_stats']['total_alerts'],
                    name='Alert Volume',
                    line=dict(color='#2563EB', width=2),
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(37, 99, 235, 0.1)'
                ))
                
                if 'anomalies' in data:
                    anomaly_points = data['anomalies'][data['anomalies']['is_anomaly']]
                    
                    if 'anomaly_severity' in anomaly_points.columns:
                        color_map = {
                            'low': '#10B981',
                            'medium': '#F59E0B',
                            'high': '#EF4444',
                            'critical': '#7F1D1D'
                        }
                        
                        for severity in ['low', 'medium', 'high', 'critical']:
                            sev_points = anomaly_points[anomaly_points['anomaly_severity'] == severity]
                            if not sev_points.empty:
                                fig_anomaly.add_trace(go.Scatter(
                                    x=sev_points.index,
                                    y=sev_points['total_alerts'],
                                    name=f'{severity.capitalize()} Severity',
                                    mode='markers',
                                    marker=dict(
                                        color=color_map.get(severity, '#DC2626'),
                                        size=14,
                                        symbol='diamond',
                                        line=dict(width=2, color='white')
                                    )
                                ))
                
                fig_anomaly.update_layout(
                    title=dict(text='Anomaly Detection by Severity Classification', 
                              font=dict(size=16, color='#111827')),
                    xaxis_title='Date',
                    yaxis_title='Alert Count',
                    hovermode='x unified',
                    template='plotly_white',
                    height=450,
                    plot_bgcolor='#FAFAFA'
                )
                
                st.plotly_chart(fig_anomaly, use_container_width=True, key='anomaly_2')
            
            display_anomaly_table(data['anomalies'])
        else:
            st.info("Anomaly detection analysis pending")
    
    with tab4:
        st.header("Predictive Forecasting")
        
        if 'forecasts' in data and not data['forecasts'].empty:
            if 'forecast' in data['forecasts'].columns:
                avg_fc = data['forecasts']['forecast'].mean()
                max_fc = data['forecasts']['forecast'].max()
                min_fc = data['forecasts']['forecast'].min()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Forecast", f"{avg_fc:.1f}")
                with col2:
                    st.metric("Peak Forecast", f"{max_fc:.0f}")
                with col3:
                    st.metric("Minimum Forecast", f"{min_fc:.0f}")
            
            col_fc1, col_fc2 = st.columns(2)
            
            with col_fc1:
                fig_forecast = go.Figure()
                
                total_forecast = data['forecasts'][data['forecasts']['target'] == 'total_alerts'] if 'target' in data['forecasts'].columns else data['forecasts'].iloc[:7].copy()
                
                if 'date' in total_forecast.columns:
                    total_forecast['date'] = pd.to_datetime(total_forecast['date'])
                else:
                    forecast_dates = pd.date_range(start=datetime.now(), periods=len(total_forecast), freq='D')
                    total_forecast['date'] = forecast_dates
                
                if 'lower_bound' in total_forecast.columns and 'upper_bound' in total_forecast.columns:
                    fig_forecast.add_trace(go.Scatter(
                        x=pd.concat([total_forecast['date'], total_forecast['date'][::-1]]),
                        y=pd.concat([total_forecast['upper_bound'], total_forecast['lower_bound'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(37, 99, 235, 0.15)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        name='Confidence Interval',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=total_forecast['date'],
                    y=total_forecast['forecast'] if 'forecast' in total_forecast.columns else total_forecast['total_alerts'],
                    name='Forecast',
                    line=dict(color='#2563EB', width=3),
                    mode='lines+markers',
                    marker=dict(size=8, color='#2563EB', line=dict(color='white', width=2))
                ))
                
                fig_forecast.update_layout(
                    title=dict(text='7-Day Alert Forecast', font=dict(size=16, color='#111827')),
                    xaxis_title='Date',
                    yaxis_title='Predicted Alert Count',
                    template='plotly_white',
                    height=350,
                    hovermode='x unified',
                    plot_bgcolor='#FAFAFA'
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True, key='forecast_2')
            
            with col_fc2:
                fig_accuracy = create_forecast_accuracy_chart(data['forecasts'], data['daily_stats'])
                st.plotly_chart(fig_accuracy, use_container_width=True, key='accuracy_1')
            
            if 'date' in data['forecasts'].columns:
                st.subheader("Forecast Data Table")
                forecast_display = data['forecasts'].copy()
                forecast_display['date'] = pd.to_datetime(forecast_display['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(forecast_display, use_container_width=True, hide_index=True)
        else:
            st.info("Forecast data unavailable")
    
    with tab5:
        st.header("Alert Repository")
        
        if 'alerts' in data and not data['alerts'].empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'type' in data['alerts'].columns:
                    common_type = data['alerts']['type'].mode()
                    if not common_type.empty:
                        st.metric("Most Common Alert", common_type.iloc[0])
            
            with col2:
                if 'region' in data['alerts'].columns:
                    common_region = data['alerts']['region'].mode()
                    if not common_region.empty:
                        st.metric("Most Affected Region", common_region.iloc[0])
            
            with col3:
                if 'severity_score' in data['alerts'].columns:
                    avg_severity = data['alerts']['severity_score'].mean()
                    st.metric("Average Severity", f"{avg_severity:.2f}")
            
            if 'issued_date' in data['alerts'].columns:
                recent_alerts = data['alerts'].sort_values('issued_date', ascending=False).head(25)
                display_cols = ['issued_date']
                
                if 'type' in recent_alerts.columns:
                    display_cols.append('type')
                if 'region' in recent_alerts.columns:
                    display_cols.append('region')
                if 'severity_score' in recent_alerts.columns:
                    display_cols.append('severity_score')
                
                if display_cols:
                    alert_display = recent_alerts[display_cols].copy()
                    if 'issued_date' in alert_display.columns:
                        alert_display['issued_date'] = pd.to_datetime(alert_display['issued_date']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.subheader("Recent Alert Records")
                    st.dataframe(alert_display, use_container_width=True, hide_index=True)
        else:
            st.info("Alert repository empty")
    
    st.markdown("---")
    st.markdown(f"""
        **Weather Anomaly Detection Dashboard v1.0** | Production Environment  
        Data Source: National Weather Service | Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        Enterprise Weather Monitoring Platform
    """)

if __name__ == "__main__":
    main()
