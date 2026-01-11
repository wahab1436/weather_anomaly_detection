import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDashboard:
    """Main dashboard class for weather anomaly detection."""
    
    def __init__(self):
        self.data_dir = 'data/output'
        self.models_dir = 'models'
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_daily_data(_self) -> Optional[pd.DataFrame]:
        """Load daily alert counts."""
        try:
            filepath = os.path.join(_self.data_dir, 'daily_alert_counts.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['date'])
                return df
        except Exception as e:
            logger.error(f"Failed to load daily data: {str(e)}")
        return None
    
    @st.cache_data(ttl=3600)
    def load_anomaly_data(_self) -> Optional[pd.DataFrame]:
        """Load anomaly detection results."""
        try:
            filepath = os.path.join(_self.data_dir, 'anomaly_detection_results.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['date'])
                return df
        except Exception as e:
            logger.error(f"Failed to load anomaly data: {str(e)}")
        return None
    
    @st.cache_data(ttl=3600)
    def load_forecast_data(_self) -> Optional[pd.DataFrame]:
        """Load forecast data."""
        try:
            filepath = os.path.join(_self.data_dir, 'alert_forecasts.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['date'])
                return df
        except Exception as e:
            logger.error(f"Failed to load forecast data: {str(e)}")
        return None
    
    @st.cache_data(ttl=3600)
    def load_anomaly_insights(_self) -> Dict:
        """Load anomaly insights."""
        try:
            filepath = os.path.join(_self.data_dir, 'anomaly_insights.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load anomaly insights: {str(e)}")
        return {}
    
    @st.cache_data(ttl=3600)
    def load_forecast_insights(_self) -> Dict:
        """Load forecast insights."""
        try:
            filepath = os.path.join(_self.data_dir, 'forecast_insights.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load forecast insights: {str(e)}")
        return {}
    
    @st.cache_data(ttl=3600)
    def load_raw_alerts(_self) -> Optional[pd.DataFrame]:
        """Load raw alert data for table display."""
        try:
            filepath = 'data/raw/weather_alerts_raw.csv'
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['scraped_at'])
                # Take only recent alerts
                if 'scraped_at' in df.columns:
                    cutoff = datetime.now() - timedelta(days=7)
                    df = df[df['scraped_at'] >= cutoff]
                return df.tail(100)  # Limit to 100 most recent
        except Exception as e:
            logger.error(f"Failed to load raw alerts: {str(e)}")
        return None
    
    def create_alert_trend_chart(self, daily_df: pd.DataFrame, anomaly_df: pd.DataFrame = None) -> go.Figure:
        """Create time series chart of alerts with anomalies."""
        fig = go.Figure()
        
        # Add daily alerts line
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['total_alerts'],
            mode='lines+markers',
            name='Daily Alerts',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Add anomaly markers if available
        if anomaly_df is not None and 'is_anomaly' in anomaly_df.columns:
            anomalies = anomaly_df[anomaly_df['is_anomaly'] == 1]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies['date'],
                    y=anomalies['total_alerts'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='<b>Anomaly Detected</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Alerts: %{y}<br>' +
                                 'Score: %{customdata:.3f}<extra></extra>',
                    customdata=anomalies['anomaly_score'] if 'anomaly_score' in anomalies.columns else [0]*len(anomalies)
                ))
        
        fig.update_layout(
            title='Daily Weather Alerts with Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Number of Alerts',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_alert_type_chart(self, daily_df: pd.DataFrame) -> go.Figure:
        """Create stacked area chart of alert types."""
        # Identify alert type columns
        alert_type_cols = [col for col in daily_df.columns if 'alert_type' in str(col)]
        
        if not alert_type_cols:
            # Create empty figure
            fig = go.Figure()
            fig.update_layout(
                title='Alert Types Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Alerts',
                template='plotly_white',
                height=400
            )
            return fig
        
        # Prepare data for stacked area
        fig = go.Figure()
        
        for col in alert_type_cols[:10]:  # Limit to top 10 types
            alert_type = col.replace('alert_type_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df[col],
                mode='lines',
                name=alert_type,
                stackgroup='one'
            ))
        
        fig.update_layout(
            title='Alert Types Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Alerts',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_forecast_chart(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
        """Create chart with historical data and forecasts."""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['total_alerts'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_alerts'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title='7-Day Alert Forecast',
            xaxis_title='Date',
            yaxis_title='Number of Alerts',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_summary_metrics(self, daily_df: pd.DataFrame) -> Dict:
        """Calculate summary metrics for display."""
        metrics = {}
        
        if daily_df.empty:
            return metrics
        
        # Today's date
        today = datetime.now().date()
        
        # Find today's data or most recent
        daily_df['date_only'] = pd.to_datetime(daily_df['date']).dt.date
        recent_data = daily_df[daily_df['date_only'] == today]
        
        if recent_data.empty:
            recent_data = daily_df.iloc[-1:]
        
        # Current alerts
        metrics['current_alerts'] = int(recent_data['total_alerts'].iloc[0])
        
        # Yesterday's alerts for comparison
        if len(daily_df) > 1:
            yesterday_data = daily_df.iloc[-2]
            metrics['yesterday_alerts'] = int(yesterday_data['total_alerts'])
            metrics['change_percent'] = ((metrics['current_alerts'] - metrics['yesterday_alerts']) / 
                                       max(metrics['yesterday_alerts'], 1)) * 100
        else:
            metrics['yesterday_alerts'] = 0
            metrics['change_percent'] = 0
        
        # 7-day average
        if len(daily_df) >= 7:
            metrics['week_avg'] = daily_df['total_alerts'].tail(7).mean()
            metrics['week_high'] = daily_df['total_alerts'].tail(7).max()
        else:
            metrics['week_avg'] = daily_df['total_alerts'].mean()
            metrics['week_high'] = daily_df['total_alerts'].max()
        
        # Alert type metrics
        alert_type_cols = [col for col in daily_df.columns if 'alert_type' in str(col)]
        if alert_type_cols:
            recent_row = daily_df.iloc[-1]
            alert_counts = {col.replace('alert_type_', ''): recent_row[col] 
                          for col in alert_type_cols if recent_row[col] > 0}
            metrics['top_alert_type'] = max(alert_counts.items(), key=lambda x: x[1])[0] if alert_counts else "None"
        
        return metrics
    
    def run(self):
        """Run the dashboard application."""
        # Title and description
        st.title("Weather Anomaly Detection Dashboard")
        st.markdown("""
        This dashboard monitors weather alert patterns, detects anomalies, and forecasts future alerts 
        using data from official weather sources.
        """)
        
        # Load data
        with st.spinner("Loading data..."):
            daily_df = self.load_daily_data()
            anomaly_df = self.load_anomaly_data()
            forecast_df = self.load_forecast_data()
            anomaly_insights = self.load_anomaly_insights()
            forecast_insights = self.load_forecast_insights()
            raw_alerts = self.load_raw_alerts()
        
        if daily_df is None:
            st.error("No data available. Please run the data collection pipeline first.")
            return
        
        # Last updated timestamp
        data_file = 'data/output/daily_alert_counts.csv'
        if os.path.exists(data_file):
            mtime = os.path.getmtime(data_file)
            last_updated = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.caption(f"Last updated: {last_updated}")
        
        # Summary metrics row
        st.subheader("Current Status")
        metrics = self.create_summary_metrics(daily_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = f"{metrics.get('change_percent', 0):.1f}%"
            delta_color = "normal" if metrics.get('change_percent', 0) == 0 else (
                "inverse" if metrics.get('change_percent', 0) > 0 else "normal"
            )
            st.metric(
                label="Today's Alerts",
                value=metrics.get('current_alerts', 0),
                delta=delta,
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                label="7-Day Average",
                value=f"{metrics.get('week_avg', 0):.0f}"
            )
        
        with col3:
            st.metric(
                label="7-Day High",
                value=f"{metrics.get('week_high', 0):.0f}"
            )
        
        with col4:
            st.metric(
                label="Top Alert Type",
                value=metrics.get('top_alert_type', 'None')
            )
        
        # Charts section
        st.subheader("Alert Trends and Anomalies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_trend = self.create_alert_trend_chart(daily_df, anomaly_df)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            fig_types = self.create_alert_type_chart(daily_df)
            st.plotly_chart(fig_types, use_container_width=True)
        
        # Forecast section
        st.subheader("Alert Forecast")
        
        if forecast_df is not None and not forecast_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_forecast = self.create_forecast_chart(daily_df, forecast_df)
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with col2:
                st.markdown("#### Forecast Details")
                for _, row in forecast_df.iterrows():
                    st.write(f"**{row['date'].strftime('%a, %b %d')}**: {int(row['predicted_alerts'])} alerts")
                    st.progress(min(row['predicted_alerts'] / 50, 1.0))
        else:
            st.info("Forecast data not available. More historical data is needed for accurate forecasting.")
        
        # Insights section
        st.subheader("Plain-English Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Anomaly Detection Insights")
            if anomaly_insights:
                insight_text = anomaly_insights.get('summary', 'No insights available.')
                st.write(insight_text)
                
                if anomaly_insights.get('key_anomalies'):
                    st.markdown("**Recent Anomalies:**")
                    for anomaly in anomaly_insights['key_anomalies'][:3]:
                        st.write(f"- {anomaly.get('date')}: {anomaly.get('reason')}")
                
                if anomaly_insights.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in anomaly_insights['recommendations']:
                        st.write(f"- {rec}")
            else:
                st.info("Anomaly insights will appear here after analysis.")
        
        with col2:
            st.markdown("#### Forecast Insights")
            if forecast_insights:
                insight_text = forecast_insights.get('summary', 'No forecast insights available.')
                st.write(insight_text)
                
                if forecast_insights.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in forecast_insights['recommendations']:
                        st.write(f"- {rec}")
            else:
                st.info("Forecast insights will appear here after analysis.")
        
        # Recent alerts table
        st.subheader("Recent Weather Alerts")
        
        if raw_alerts is not None and not raw_alerts.empty:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alert_types = ['All'] + sorted(raw_alerts['alert_type'].unique().tolist())
                selected_type = st.selectbox("Filter by Alert Type", alert_types)
            
            with col2:
                regions = ['All'] + sorted(raw_alerts['region'].unique().tolist())
                selected_region = st.selectbox("Filter by Region", regions)
            
            with col3:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now().date() - timedelta(days=3), datetime.now().date()),
                    max_value=datetime.now().date()
                )
            
            # Apply filters
            filtered_df = raw_alerts.copy()
            
            if selected_type != 'All':
                filtered_df = filtered_df[filtered_df['alert_type'] == selected_type]
            
            if selected_region != 'All':
                filtered_df = filtered_df[filtered_df['region'] == selected_region]
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['scraped_at'].dt.date >= start_date) &
                    (filtered_df['scraped_at'].dt.date <= end_date)
                ]
            
            # Display table
            display_cols = ['scraped_at', 'alert_type', 'region', 'title', 'severity']
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_cols].sort_values('scraped_at', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Export option
            if st.button("Export Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="filtered_alerts.csv",
                    mime="text/csv"
                )
        else:
            st.info("No recent alerts available. Data collection may be in progress.")
        
        # Sidebar controls
        with st.sidebar:
            st.markdown("### Dashboard Controls")
            
            # Data freshness indicator
            st.markdown("#### Data Status")
            if daily_df is not None:
                latest_date = daily_df['date'].max()
                days_old = (datetime.now().date() - latest_date.date()).days
                
                if days_old == 0:
                    st.success("Data is current (today)")
                elif days_old == 1:
                    st.info("Data is from yesterday")
                elif days_old <= 3:
                    st.warning(f"Data is {days_old} days old")
                else:
                    st.error(f"Data is {days_old} days old - may need update")
            
            # Manual refresh
            if st.button("Force Refresh Data", type="secondary"):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""
            This dashboard provides:
            
            - Real-time weather alert monitoring
            - Anomaly detection using ML
            - 7-day alert forecasts
            - Plain-English insights
            
            Data source: National Weather Service
            Update frequency: Hourly
            """)

def main():
    """Main function to run the dashboard."""
    dashboard = WeatherDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
