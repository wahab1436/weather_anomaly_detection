"""
Main Streamlit dashboard for Weather Anomaly Detection
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from src.utils.helpers import load_data, validate_dataframe

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .anomaly-box {
        background-color: #fde8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }
    .forecast-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    .header-text {
        color: #343a40;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

class WeatherAnomalyDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_loaded = False
        self.df_alerts = None
        self.df_anomalies = None
        self.df_forecast = None
        self.last_updated = None
        self.load_data()
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_data(_self):
        """Load all necessary data"""
        try:
            # Load processed alerts
            if os.path.exists(Config.PROCESSED_DATA_PATH):
                _self.df_alerts = pd.read_csv(Config.PROCESSED_DATA_PATH)
                if 'timestamp' in _self.df_alerts.columns:
                    _self.df_alerts['timestamp'] = pd.to_datetime(_self.df_alerts['timestamp'])
                print(f"Loaded alerts: {len(_self.df_alerts)} rows")
            else:
                _self.df_alerts = pd.DataFrame()
                print("No alerts data found")
            
            # Load anomaly results
            if os.path.exists(Config.ANOMALY_OUTPUT_PATH):
                _self.df_anomalies = pd.read_csv(Config.ANOMALY_OUTPUT_PATH, index_col=0)
                if _self.df_anomalies.index.name == 'date':
                    _self.df_anomalies.index = pd.to_datetime(_self.df_anomalies.index)
                print(f"Loaded anomalies: {len(_self.df_anomalies)} rows")
            else:
                _self.df_anomalies = pd.DataFrame()
                print("No anomaly data found")
            
            # Load forecast results
            if os.path.exists(Config.FORECAST_OUTPUT_PATH):
                _self.df_forecast = pd.read_csv(Config.FORECAST_OUTPUT_PATH)
                if 'date' in _self.df_forecast.columns:
                    _self.df_forecast['date'] = pd.to_datetime(_self.df_forecast['date'])
                print(f"Loaded forecast: {len(_self.df_forecast)} rows")
            else:
                _self.df_forecast = pd.DataFrame()
                print("No forecast data found")
            
            _self.data_loaded = True
            _self.last_updated = datetime.now()
            
            # Validate data
            if not _self.df_alerts.empty:
                validation = validate_dataframe(_self.df_alerts)
                if not validation['is_valid']:
                    print(f"Data validation issues: {validation['issues']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            _self.data_loaded = False
            return False
    
    def get_summary_metrics(self) -> Dict:
        """Calculate summary metrics"""
        metrics = {
            'total_alerts': 0,
            'recent_alerts': 0,
            'anomaly_count': 0,
            'top_region': 'N/A',
            'top_alert_type': 'N/A',
            'forecast_avg': 0
        }
        
        if not self.data_loaded:
            return metrics
        
        # Total alerts
        if not self.df_alerts.empty:
            metrics['total_alerts'] = len(self.df_alerts)
            
            # Recent alerts (last 7 days)
            if 'timestamp' in self.df_alerts.columns:
                recent_cutoff = datetime.now() - timedelta(days=7)
                recent_alerts = self.df_alerts[self.df_alerts['timestamp'] > recent_cutoff]
                metrics['recent_alerts'] = len(recent_alerts)
            
            # Top region
            if 'region' in self.df_alerts.columns:
                region_counts = self.df_alerts['region'].value_counts()
                if not region_counts.empty:
                    metrics['top_region'] = region_counts.index[0]
            
            # Top alert type
            if 'alert_type' in self.df_alerts.columns:
                type_counts = self.df_alerts['alert_type'].value_counts()
                if not type_counts.empty:
                    metrics['top_alert_type'] = type_counts.index[0]
        
        # Anomaly count
        if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
            metrics['anomaly_count'] = int(self.df_anomalies['is_anomaly'].sum())
        
        # Forecast average
        if not self.df_forecast.empty and 'forecast' in self.df_forecast.columns:
            metrics['forecast_avg'] = float(self.df_forecast['forecast'].mean())
        
        return metrics
    
    def create_alert_trend_chart(self) -> Optional[go.Figure]:
        """Create alert trend chart with anomalies"""
        if self.df_anomalies.empty:
            return None
        
        fig = go.Figure()
        
        # Add alert trend line
        if 'total_alerts' in self.df_anomalies.columns:
            fig.add_trace(go.Scatter(
                x=self.df_anomalies.index,
                y=self.df_anomalies['total_alerts'],
                mode='lines',
                name='Daily Alerts',
                line=dict(color='#007bff', width=2),
                hovertemplate='Date: %{x}<br>Alerts: %{y}<extra></extra>'
            ))
        
        # Add anomaly markers
        if 'is_anomaly' in self.df_anomalies.columns:
            anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']]
            if not anomalies.empty and 'total_alerts' in anomalies.columns:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies['total_alerts'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='#dc3545',
                        size=10,
                        symbol='diamond',
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='Date: %{x}<br>Alerts: %{y}<br>Anomaly Detected<extra></extra>'
                ))
        
        # Add forecast if available
        if not self.df_forecast.empty:
            fig.add_trace(go.Scatter(
                x=self.df_forecast['date'],
                y=self.df_forecast['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='#28a745', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Forecast: %{y}<extra></extra>'
            ))
            
            # Add confidence interval
            if 'lower_bound' in self.df_forecast.columns and 'upper_bound' in self.df_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([self.df_forecast['date'], self.df_forecast['date'][::-1]]),
                    y=pd.concat([self.df_forecast['upper_bound'], self.df_forecast['lower_bound'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(40, 167, 69, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='Confidence Interval'
                ))
        
        fig.update_layout(
            title=dict(
                text='Weather Alert Trends with Anomaly Detection',
                font=dict(size=16, color='#343a40')
            ),
            xaxis=dict(
                title='Date',
                gridcolor='#e9ecef',
                showgrid=True
            ),
            yaxis=dict(
                title='Number of Alerts',
                gridcolor='#e9ecef',
                showgrid=True
            ),
            hovermode='x unified',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#495057')
        )
        
        return fig
    
    def create_alert_type_chart(self) -> Optional[go.Figure]:
        """Create alert type distribution chart"""
        if self.df_alerts.empty or 'alert_type' not in self.df_alerts.columns:
            return None
        
        # Get alert type counts
        alert_counts = self.df_alerts['alert_type'].value_counts().reset_index()
        alert_counts.columns = ['alert_type', 'count']
        
        # Sort by count
        alert_counts = alert_counts.sort_values('count', ascending=False)
        
        fig = px.bar(
            alert_counts,
            x='alert_type',
            y='count',
            color='alert_type',
            title='Alert Type Distribution',
            height=400,
            labels={'alert_type': 'Alert Type', 'count': 'Count'}
        )
        
        fig.update_layout(
            xaxis_title='Alert Type',
            yaxis_title='Count',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#495057')
        )
        
        return fig
    
    def create_region_analysis_chart(self) -> Optional[go.Figure]:
        """Create region analysis chart"""
        if self.df_alerts.empty or 'region' not in self.df_alerts.columns:
            return None
        
        # Get region counts
        region_counts = self.df_alerts['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        
        # Limit to top 10 regions
        region_counts = region_counts.head(10)
        
        fig = px.bar(
            region_counts,
            x='region',
            y='count',
            color='region',
            title='Top Regions by Alert Count',
            height=400,
            labels={'region': 'Region', 'count': 'Alert Count'}
        )
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Alert Count',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#495057')
        )
        
        return fig
    
    def create_severity_chart(self) -> Optional[go.Figure]:
        """Create severity distribution chart"""
        if self.df_alerts.empty or 'severity' not in self.df_alerts.columns:
            return None
        
        # Get severity counts
        severity_counts = self.df_alerts['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        
        fig = px.pie(
            severity_counts,
            values='count',
            names='severity',
            title='Alert Severity Distribution',
            height=400
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#495057')
        )
        
        return fig
    
    def create_anomaly_analysis_chart(self) -> Optional[go.Figure]:
        """Create anomaly analysis chart"""
        if self.df_anomalies.empty or 'anomaly_score' not in self.df_anomalies.columns:
            return None
        
        fig = go.Figure()
        
        # Add anomaly scores
        fig.add_trace(go.Scatter(
            x=self.df_anomalies.index,
            y=self.df_anomalies['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#6f42c1', width=2),
            hovertemplate='Date: %{x}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        # Highlight anomalies
        if 'is_anomaly' in self.df_anomalies.columns:
            anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies['anomaly_score'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color='#dc3545',
                        size=8,
                        symbol='x'
                    ),
                    hovertemplate='Date: %{x}<br>Score: %{y:.3f}<br>ANOMALY<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text='Anomaly Score Timeline',
                font=dict(size=16, color='#343a40')
            ),
            xaxis=dict(
                title='Date',
                gridcolor='#e9ecef',
                showgrid=True
            ),
            yaxis=dict(
                title='Anomaly Score',
                gridcolor='#e9ecef',
                showgrid=True
            ),
            hovermode='x unified',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#495057')
        )
        
        return fig
    
    def generate_insights(self) -> List[Dict]:
        """Generate plain-English insights"""
        insights = []
        
        if not self.data_loaded:
            insights.append({
                'type': 'info',
                'text': 'Data is currently being loaded. Please wait...'
            })
            return insights
        
        # Check if we have data
        if self.df_alerts.empty:
            insights.append({
                'type': 'warning',
                'text': 'No alert data available. Run the scraping pipeline first.'
            })
            return insights
        
        # Insight 1: Alert volume trend
        if not self.df_anomalies.empty and 'total_alerts' in self.df_anomalies.columns:
            recent_days = 7
            if len(self.df_anomalies) > recent_days:
                recent_avg = self.df_anomalies['total_alerts'].tail(recent_days).mean()
                previous_avg = self.df_anomalies['total_alerts'].head(-recent_days).mean()
                
                if previous_avg > 0:
                    change_pct = ((recent_avg - previous_avg) / previous_avg) * 100
                    
                    if abs(change_pct) > 20:
                        direction = "increased" if change_pct > 0 else "decreased"
                        insights.append({
                            'type': 'trend',
                            'text': f'Alert volume has {direction} by {abs(change_pct):.1f}% in the last {recent_days} days compared to previous period.'
                        })
        
        # Insight 2: Recent anomalies
        if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
            recent_anomalies = self.df_anomalies.tail(7)['is_anomaly'].sum()
            if recent_anomalies > 0:
                insights.append({
                    'type': 'anomaly',
                    'text': f'Detected {int(recent_anomalies)} anomaly patterns in the last 7 days requiring attention.'
                })
        
        # Insight 3: Top alert type
        if 'alert_type' in self.df_alerts.columns:
            alert_dist = self.df_alerts['alert_type'].value_counts(normalize=True)
            if not alert_dist.empty:
                top_type = alert_dist.index[0]
                top_pct = alert_dist.iloc[0] * 100
                
                if top_pct > 25:
                    insights.append({
                        'type': 'distribution',
                        'text': f'{top_type.capitalize()} alerts account for {top_pct:.1f}% of all weather alerts.'
                    })
        
        # Insight 4: Forecast insight
        if not self.df_forecast.empty and 'forecast' in self.df_forecast.columns:
            avg_forecast = self.df_forecast['forecast'].mean()
            
            if not self.df_anomalies.empty and 'total_alerts' in self.df_anomalies.columns:
                recent_avg = self.df_anomalies['total_alerts'].tail(7).mean()
                
                if avg_forecast > recent_avg * 1.3:
                    insights.append({
                        'type': 'forecast',
                        'text': 'Forecast indicates elevated alert activity expected in the coming days.'
                    })
                elif avg_forecast < recent_avg * 0.7:
                    insights.append({
                        'type': 'forecast',
                        'text': 'Forecast indicates reduced alert activity expected in the coming days.'
                    })
        
        # Insight 5: Time patterns
        if 'hour' in self.df_alerts.columns:
            # Find peak hour
            hour_counts = self.df_alerts['hour'].value_counts()
            if not hour_counts.empty:
                peak_hour = hour_counts.index[0]
                
                # Check if there's a clear peak
                if hour_counts.iloc[0] > hour_counts.iloc[1] * 1.5:
                    insights.append({
                        'type': 'pattern',
                        'text': f'Peak alert issuance occurs at {int(peak_hour):02d}:00 UTC.'
                    })
        
        # Insight 6: Regional insight
        if 'region' in self.df_alerts.columns and 'alert_type' in self.df_alerts.columns:
            # Find region with most specific alert type
            for alert_type in ['flood', 'storm', 'fire']:
                if alert_type in self.df_alerts['alert_type'].values:
                    type_data = self.df_alerts[self.df_alerts['alert_type'] == alert_type]
                    if not type_data.empty and 'region' in type_data.columns:
                        top_region = type_data['region'].value_counts()
                        if not top_region.empty:
                            insights.append({
                                'type': 'regional',
                                'text': f'{alert_type.capitalize()} alerts are most frequent in the {top_region.index[0]} region.'
                            })
                            break
        
        # If no insights generated, add a general one
        if not insights:
            insights.append({
                'type': 'info',
                'text': 'No significant patterns detected in current data. Monitoring continues.'
            })
        
        return insights
    
    def display_header(self):
        """Display dashboard header"""
        st.markdown("<h1 class='header-text'>Weather Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("Real-time monitoring and forecasting of weather alert anomalies")
        st.markdown("---")
        
        # Last updated timestamp
        if self.last_updated:
            st.caption(f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    def display_metrics_row(self):
        """Display summary metrics in a row"""
        metrics = self.get_summary_metrics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Alerts",
                value=f"{metrics['total_alerts']:,}",
                help="Total number of weather alerts collected"
            )
        
        with col2:
            st.metric(
                label="7-Day Alerts",
                value=f"{metrics['recent_alerts']:,}",
                help="Alerts from the last 7 days"
            )
        
        with col3:
            st.metric(
                label="Anomalies",
                value=f"{metrics['anomaly_count']}",
                help="Total anomalies detected"
            )
        
        with col4:
            st.metric(
                label="Top Region",
                value=metrics['top_region'],
                help="Region with most alerts"
            )
        
        with col5:
            st.metric(
                label="Avg Forecast",
                value=f"{metrics['forecast_avg']:.1f}",
                help="Average forecasted alerts per day"
            )
        
        st.markdown("---")
    
    def display_insights_panel(self):
        """Display insights panel"""
        st.subheader("Key Insights")
        
        insights = self.generate_insights()
        
        for insight in insights:
            insight_type = insight['type']
            text = insight['text']
            
            if insight_type == 'anomaly':
                st.markdown(f"""
                    <div class="anomaly-box">
                        <strong>Anomaly Detection:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            elif insight_type == 'forecast':
                st.markdown(f"""
                    <div class="forecast-box">
                        <strong>Forecast Insight:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>Analysis:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
    
    def display_main_charts(self):
        """Display main charts in columns"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Alert trend chart
            trend_chart = self.create_alert_trend_chart()
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                st.info("Trend chart data not available. Run the anomaly detection pipeline.")
        
        with col2:
            # Insights panel
            self.display_insights_panel()
        
        st.markdown("---")
    
    def display_secondary_charts(self):
        """Display secondary charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert type chart
            type_chart = self.create_alert_type_chart()
            if type_chart:
                st.plotly_chart(type_chart, use_container_width=True)
            else:
                st.info("Alert type data not available")
        
        with col2:
            # Region analysis chart
            region_chart = self.create_region_analysis_chart()
            if region_chart:
                st.plotly_chart(region_chart, use_container_width=True)
            else:
                st.info("Region data not available")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Anomaly analysis chart
            anomaly_chart = self.create_anomaly_analysis_chart()
            if anomaly_chart:
                st.plotly_chart(anomaly_chart, use_container_width=True)
            else:
                st.info("Anomaly analysis data not available")
        
        with col4:
            # Severity chart
            severity_chart = self.create_severity_chart()
            if severity_chart:
                st.plotly_chart(severity_chart, use_container_width=True)
            else:
                st.info("Severity data not available")
        
        st.markdown("---")
    
    def display_data_tables(self):
        """Display data tables"""
        tab1, tab2, tab3 = st.tabs(["Recent Alerts", "Anomalies", "Forecast"])
        
        with tab1:
            if not self.df_alerts.empty:
                # Show recent alerts
                recent_alerts = self.df_alerts.sort_values('timestamp', ascending=False).head(20)
                
                # Select columns to display
                display_cols = ['timestamp', 'title', 'region', 'alert_type', 'severity']
                display_cols = [col for col in display_cols if col in recent_alerts.columns]
                
                if display_cols:
                    st.dataframe(
                        recent_alerts[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'timestamp': st.column_config.DatetimeColumn('Timestamp'),
                            'title': st.column_config.TextColumn('Alert'),
                            'region': st.column_config.TextColumn('Region'),
                            'alert_type': st.column_config.TextColumn('Type'),
                            'severity': st.column_config.TextColumn('Severity')
                        }
                    )
            else:
                st.info("No alert data available")
        
        with tab2:
            if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
                # Show anomalies
                anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']].sort_index(ascending=False)
                
                # Select columns to display
                display_cols = ['total_alerts', 'anomaly_score']
                if 'anomaly_probability' in anomalies.columns:
                    display_cols.append('anomaly_probability')
                
                # Add any alert type columns that exist
                for alert_type in Config.ALERT_TYPES.keys():
                    if alert_type in anomalies.columns:
                        display_cols.append(alert_type)
                
                display_cols = [col for col in display_cols if col in anomalies.columns]
                
                if display_cols:
                    # Format the dataframe
                    display_df = anomalies[display_cols].copy()
                    display_df.index.name = 'date'
                    
                    if 'anomaly_probability' in display_df.columns:
                        display_df['anomaly_probability'] = display_df['anomaly_probability'].apply(
                            lambda x: f"{x:.1%}"
                        )
                    
                    st.dataframe(
                        display_df.head(20),
                        use_container_width=True
                    )
            else:
                st.info("No anomaly data available")
        
        with tab3:
            if not self.df_forecast.empty:
                # Show forecast
                display_df = self.df_forecast.copy()
                
                # Format date
                if 'date' in display_df.columns:
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                
                # Format numbers
                for col in ['forecast', 'lower_bound', 'upper_bound']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'date': st.column_config.TextColumn('Date'),
                        'forecast': st.column_config.TextColumn('Forecast'),
                        'lower_bound': st.column_config.TextColumn('Lower Bound'),
                        'upper_bound': st.column_config.TextColumn('Upper Bound')
                    }
                )
            else:
                st.info("No forecast data available")
    
    def display_footer(self):
        """Display dashboard footer"""
        st.markdown("---")
        
        # Data source and info
        st.markdown("""
            <div style="color: #6c757d; font-size: 0.9rem;">
                <p><strong>Data Source:</strong> National Weather Service (weather.gov)</p>
                <p><strong>Update Frequency:</strong> Hourly scraping, real-time processing</p>
                <p><strong>Anomaly Detection:</strong> Isolation Forest with statistical methods</p>
                <p><strong>Forecast Model:</strong> XGBoost time series forecasting</p>
                <p><strong>Dashboard Version:</strong> 1.0.0 | Production Ready</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.caption("""
            **Disclaimer:** This dashboard is for monitoring and analysis purposes only. 
            Always refer to official weather sources for critical weather information.
        """)
    
    def run(self):
        """Run the dashboard"""
        # Display header
        self.display_header()
        
        # Data status warning
        if not self.data_loaded:
            st.warning("Dashboard data is not fully available. Please ensure the data pipeline is running.")
            
            # Show setup instructions
            with st.expander("Setup Instructions"):
                st.markdown("""
                    1. **Install dependencies:**
                    ```bash
                    pip install -r requirements.txt
                    ```
                    
                    2. **Run the complete pipeline:**
                    ```bash
                    python run_dashboard.py --mode pipeline
                    ```
                    
                    3. **Start the dashboard:**
                    ```bash
                    python run_dashboard.py --mode dashboard
                    ```
                    
                    4. **Schedule hourly scraping (optional):**
                    ```bash
                    python run_dashboard.py --schedule
                    ```
                """)
            return
        
        # Display metrics
        self.display_metrics_row()
        
        # Display main content
        self.display_main_charts()
        self.display_secondary_charts()
        self.display_data_tables()
        
        # Display footer
        self.display_footer()

def main():
    """Main function to run the dashboard"""
    # Initialize dashboard
    dashboard = WeatherAnomalyDashboard()
    
    # Run dashboard
    dashboard.run()

if __name__ == "__main__":
    main()
