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
from typing import Dict, List, Optional, Tuple, Any

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
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
    .data-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
        self.setup_directories()
        self.load_data()
    
    def setup_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            'data/raw',
            'data/processed',
            'data/output',
            'models',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @st.cache_data(ttl=300)
    def load_data(_self):
        """Load all necessary data"""
        try:
            from config import Config
            
            # Check if data files exist
            data_exists = (
                os.path.exists(Config.PROCESSED_DATA_PATH) and
                os.path.exists(Config.ANOMALY_OUTPUT_PATH) and
                os.path.exists(Config.FORECAST_OUTPUT_PATH)
            )
            
            if not data_exists:
                return False
            
            # Load processed alerts
            try:
                _self.df_alerts = pd.read_csv(Config.PROCESSED_DATA_PATH)
                if 'timestamp' in _self.df_alerts.columns:
                    _self.df_alerts['timestamp'] = pd.to_datetime(_self.df_alerts['timestamp'])
            except Exception:
                _self.df_alerts = pd.DataFrame()
            
            # Load anomaly results
            try:
                _self.df_anomalies = pd.read_csv(Config.ANOMALY_OUTPUT_PATH, index_col=0)
                if _self.df_anomalies.index.name == 'date':
                    _self.df_anomalies.index = pd.to_datetime(_self.df_anomalies.index)
            except Exception:
                _self.df_anomalies = pd.DataFrame()
            
            # Load forecast results
            try:
                _self.df_forecast = pd.read_csv(Config.FORECAST_OUTPUT_PATH)
                if 'date' in _self.df_forecast.columns:
                    _self.df_forecast['date'] = pd.to_datetime(_self.df_forecast['date'])
            except Exception:
                _self.df_forecast = pd.DataFrame()
            
            _self.data_loaded = True
            _self.last_updated = datetime.now()
            
            return True
            
        except Exception:
            return False
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics"""
        metrics = {
            'total_alerts': 0,
            'recent_alerts': 0,
            'anomaly_count': 0,
            'top_region': 'N/A',
            'top_alert_type': 'N/A',
            'forecast_avg': 0,
            'data_days': 0
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
                metrics['data_days'] = (self.df_alerts['timestamp'].max() - self.df_alerts['timestamp'].min()).days
            
            # Top region
            if 'region' in self.df_alerts.columns:
                region_counts = self.df_alerts['region'].value_counts()
                if not region_counts.empty:
                    metrics['top_region'] = region_counts.index[0]
            
            # Top alert type
            if 'alert_type' in self.df_alerts.columns:
                type_counts = self.df_alerts['alert_type'].value_counts()
                if not type_counts.empty:
                    metrics['top_alert_type'] = type_counts.index[0].capitalize()
        
        # Anomaly count
        if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
            metrics['anomaly_count'] = int(self.df_anomalies['is_anomaly'].sum())
        
        # Forecast average
        if not self.df_forecast.empty and 'forecast' in self.df_forecast.columns:
            metrics['forecast_avg'] = float(self.df_forecast['forecast'].mean())
        
        return metrics
    
    def create_alert_trend_chart(self) -> Optional[go.Figure]:
        """Create alert trend chart with anomalies"""
        if self.df_anomalies.empty or 'total_alerts' not in self.df_anomalies.columns:
            return None
        
        fig = go.Figure()
        
        # Add alert trend line
        fig.add_trace(go.Scatter(
            x=self.df_anomalies.index,
            y=self.df_anomalies['total_alerts'],
            mode='lines',
            name='Daily Alerts',
            line=dict(color='#007bff', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Alerts: %{y}<extra></extra>'
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
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Alerts: %{y}<br>Anomaly Detected<extra></extra>'
                ))
        
        # Add forecast if available
        if not self.df_forecast.empty:
            fig.add_trace(go.Scatter(
                x=self.df_forecast['date'],
                y=self.df_forecast['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='#28a745', width=2, dash='dash'),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra></extra>'
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
            title='Weather Alert Trends with Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Number of Alerts',
            hovermode='x unified',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
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
        
        # Format alert type names
        alert_counts['alert_type'] = alert_counts['alert_type'].str.capitalize()
        
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
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_region_chart(self) -> Optional[go.Figure]:
        """Create region distribution chart"""
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
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_heatmap_chart(self) -> Optional[go.Figure]:
        """Create heatmap of alerts by hour and day"""
        if self.df_alerts.empty or 'timestamp' not in self.df_alerts.columns:
            return None
        
        # Extract hour and day of week
        df = self.df_alerts.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Create pivot table
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        
        # Create 2D array for heatmap
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = list(range(24))
        
        z_data = np.zeros((7, 24))
        for _, row in heatmap_data.iterrows():
            z_data[int(row['day_of_week']), int(row['hour'])] = row['count']
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=hours,
            y=days,
            colorscale='YlOrRd',
            hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Alerts: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Alert Distribution by Day and Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_severity_chart(self) -> Optional[go.Figure]:
        """Create severity distribution chart"""
        if self.df_alerts.empty or 'severity' not in self.df_alerts.columns:
            return None
        
        # Get severity counts
        severity_counts = self.df_alerts['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        
        # Format severity names
        severity_counts['severity'] = severity_counts['severity'].str.capitalize()
        
        fig = px.pie(
            severity_counts,
            values='count',
            names='severity',
            title='Alert Severity Distribution',
            height=350
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
    
    def create_anomaly_timeline_chart(self) -> Optional[go.Figure]:
        """Create anomaly score timeline chart"""
        if self.df_anomalies.empty or 'anomaly_score' not in self.df_anomalies.columns:
            return None
        
        fig = go.Figure()
        
        # Add anomaly scores
        fig.add_trace(go.Scatter(
            x=self.df_anomalies.index,
            y=self.df_anomalies['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#6c757d', width=1.5),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        # Highlight anomalies
        if 'is_anomaly' in self.df_anomalies.columns:
            anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']]
            if not anomalies.empty and 'anomaly_score' in anomalies.columns:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies['anomaly_score'],
                    mode='markers',
                    name='Detected Anomaly',
                    marker=dict(
                        color='#dc3545',
                        size=10,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Score: %{y:.3f}<br>Anomaly<extra></extra>'
                ))
        
        fig.update_layout(
            title='Anomaly Detection Timeline',
            xaxis_title='Date',
            yaxis_title='Anomaly Score',
            hovermode='x unified',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def generate_insights(self) -> List[Dict[str, str]]:
        """Generate plain-English insights"""
        insights = []
        
        if not self.data_loaded:
            insights.append({
                'type': 'warning',
                'text': 'Data is being loaded. Please wait.'
            })
            return insights
        
        # Check if we have data
        if self.df_alerts.empty:
            insights.append({
                'type': 'error',
                'text': 'No alert data available. Run the pipeline first using: python run_dashboard.py --pipeline'
            })
            return insights
        
        metrics = self.get_summary_metrics()
        
        # Insight 1: Alert volume
        if metrics['recent_alerts'] > 0:
            avg_daily = metrics['recent_alerts'] / 7
            insights.append({
                'type': 'info',
                'text': f'Average of {avg_daily:.1f} alerts per day in the last week.'
            })
        
        # Insight 2: Anomalies
        if metrics['anomaly_count'] > 0:
            insights.append({
                'type': 'anomaly',
                'text': f'Detected {metrics["anomaly_count"]} unusual patterns requiring attention.'
            })
        
        # Insight 3: Top alert type
        if metrics['top_alert_type'] != 'N/A':
            insights.append({
                'type': 'info',
                'text': f'Most frequent alert type: {metrics["top_alert_type"]}.'
            })
        
        # Insight 4: Forecast
        if metrics['forecast_avg'] > 0:
            insights.append({
                'type': 'forecast',
                'text': f'Forecast predicts {metrics["forecast_avg"]:.1f} alerts per day on average.'
            })
        
        # Insight 5: Data coverage
        if metrics['data_days'] > 0:
            insights.append({
                'type': 'info',
                'text': f'Data covers {metrics["data_days"]} days of historical weather alerts.'
            })
        
        # Insight 6: Regional analysis
        if metrics['top_region'] != 'N/A':
            insights.append({
                'type': 'info',
                'text': f'Highest alert activity in {metrics["top_region"]} region.'
            })
        
        # Add more insights based on data patterns
        if not self.df_anomalies.empty and len(self.df_anomalies) > 14:
            recent_avg = self.df_anomalies['total_alerts'].tail(7).mean()
            older_avg = self.df_anomalies['total_alerts'].head(-7).mean()
            
            if older_avg > 0:
                change = ((recent_avg - older_avg) / older_avg) * 100
                if abs(change) > 15:
                    direction = "increased" if change > 0 else "decreased"
                    insights.append({
                        'type': 'trend',
                        'text': f'Alert volume has {direction} by {abs(change):.1f}% compared to previous period.'
                    })
        
        return insights
    
    def display_header(self):
        """Display dashboard header"""
        st.title("Weather Anomaly Detection Dashboard")
        st.markdown("Real-time monitoring and forecasting of weather alert anomalies")
        
        if self.last_updated:
            st.caption(f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
    
    def display_data_status(self):
        """Display data status panel"""
        if not self.data_loaded:
            st.warning("Dashboard data is not available. Please run the data pipeline first.")
            
            with st.expander("Setup Instructions"):
                st.markdown("""
                **Run these commands in your terminal:**

                1. **Install dependencies:**
                ```
                pip install -r requirements.txt
                ```

                2. **Create directories:**
                ```
                mkdir -p data/raw data/processed data/output models logs
                ```

                3. **Run the complete pipeline (this will take a few minutes):**
                ```
                python run_dashboard.py --pipeline
                ```

                4. **Refresh this dashboard** after the pipeline completes.

                ---

                **Alternative: Run individual steps**
                ```
                # Just scrape data
                python run_dashboard.py --scrape

                # Just process data
                python run_dashboard.py --process

                # Just run anomaly detection
                python run_dashboard.py --anomaly

                # Just run forecasting
                python run_dashboard.py --forecast
                ```
                """)
            
            return False
        
        return True
    
    def display_metrics(self):
        """Display summary metrics"""
        metrics = self.get_summary_metrics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Alerts",
                value=f"{metrics['total_alerts']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="7-Day Alerts",
                value=f"{metrics['recent_alerts']:,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Anomalies",
                value=f"{metrics['anomaly_count']}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Top Region",
                value=metrics['top_region'],
                delta=None
            )
        
        with col5:
            st.metric(
                label="Forecast Avg",
                value=f"{metrics['forecast_avg']:.1f}",
                delta=None
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
                        <strong>Anomaly Alert:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            elif insight_type == 'forecast':
                st.markdown(f"""
                    <div class="forecast-box">
                        <strong>Forecast:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            elif insight_type == 'warning':
                st.markdown(f"""
                    <div class="anomaly-box">
                        <strong>Warning:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            elif insight_type == 'error':
                st.markdown(f"""
                    <div class="anomaly-box">
                        <strong>Error:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            elif insight_type == 'trend':
                st.markdown(f"""
                    <div class="forecast-box">
                        <strong>Trend:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>Insight:</strong> {text}
                    </div>
                """, unsafe_allow_html=True)
    
    def display_main_charts(self):
        """Display main charts"""
        # Row 1: Trend chart and insights
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trend_chart = self.create_alert_trend_chart()
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                st.info("Trend chart data not available. Run anomaly detection.")
        
        with col2:
            self.display_insights_panel()
        
        st.markdown("---")
        
        # Row 2: Distribution charts
        col3, col4 = st.columns(2)
        
        with col3:
            type_chart = self.create_alert_type_chart()
            if type_chart:
                st.plotly_chart(type_chart, use_container_width=True)
            else:
                st.info("Alert type data not available")
        
        with col4:
            region_chart = self.create_region_chart()
            if region_chart:
                st.plotly_chart(region_chart, use_container_width=True)
            else:
                st.info("Region data not available")
        
        # Row 3: Analysis charts
        col5, col6 = st.columns(2)
        
        with col5:
            anomaly_chart = self.create_anomaly_timeline_chart()
            if anomaly_chart:
                st.plotly_chart(anomaly_chart, use_container_width=True)
            else:
                st.info("Anomaly timeline data not available")
        
        with col6:
            severity_chart = self.create_severity_chart()
            if severity_chart:
                st.plotly_chart(severity_chart, use_container_width=True)
            else:
                st.info("Severity data not available")
        
        st.markdown("---")
    
    def display_data_tables(self):
        """Display data tables in tabs"""
        tab1, tab2, tab3, tab4 = st.tabs(["Recent Alerts", "Anomalies", "Forecast", "System Info"])
        
        with tab1:
            if not self.df_alerts.empty:
                # Show recent alerts
                recent_alerts = self.df_alerts.sort_values('timestamp', ascending=False).head(50)
                
                # Select columns to display
                display_cols = []
                for col in ['timestamp', 'title', 'region', 'alert_type', 'severity']:
                    if col in recent_alerts.columns:
                        display_cols.append(col)
                
                if display_cols:
                    st.dataframe(
                        recent_alerts[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No alert columns available for display")
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
                
                # Add any alert type columns
                for col in anomalies.columns:
                    if col in ['flood', 'storm', 'fire', 'wind', 'heat', 'cold', 'winter']:
                        display_cols.append(col)
                
                display_cols = [col for col in display_cols if col in anomalies.columns]
                
                if display_cols:
                    # Format the dataframe
                    display_df = anomalies[display_cols].copy()
                    display_df.index.name = 'date'
                    
                    if 'anomaly_probability' in display_df.columns:
                        display_df['anomaly_probability'] = (display_df['anomaly_probability'] * 100).round(1)
                    
                    st.dataframe(
                        display_df.head(20),
                        use_container_width=True
                    )
                else:
                    st.info("No anomaly columns available for display")
            else:
                st.info("No anomaly data available")
        
        with tab3:
            if not self.df_forecast.empty:
                # Show forecast
                display_df = self.df_forecast.copy()
                
                # Format numbers
                for col in ['forecast', 'lower_bound', 'upper_bound']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(1)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No forecast data available")
        
        with tab4:
            # System information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Sources")
                st.markdown("""
                - Primary: weather.gov/alerts
                - Supplementary: weather.gov/wrh/TextProduct
                - Update Frequency: Hourly
                - Data Type: Unstructured textual alerts
                """)
            
            with col2:
                st.subheader("ML Models")
                st.markdown("""
                - Anomaly Detection: Isolation Forest algorithm
                - Forecasting: XGBoost time series model
                - Feature Engineering: Text analysis + temporal features
                - Output: Daily predictions with confidence intervals
                """)
            
            st.subheader("System Status")
            
            # Check file status
            from config import Config
            
            files = [
                ("Raw Alerts", Config.RAW_DATA_PATH),
                ("Processed Data", Config.PROCESSED_DATA_PATH),
                ("Anomaly Results", Config.ANOMALY_OUTPUT_PATH),
                ("Forecast Results", Config.FORECAST_OUTPUT_PATH),
                ("Anomaly Model", Config.ANOMALY_MODEL_PATH),
                ("Forecast Model", Config.FORECAST_MODEL_PATH)
            ]
            
            for file_name, file_path in files:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / 1024
                    st.success(f"{file_name}: {size:.1f} KB")
                else:
                    st.error(f"{file_name}: Not found")
    
    def display_footer(self):
        """Display dashboard footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                **Data Source**  
                National Weather Service  
                Public weather alerts API
            """)
        
        with col2:
            st.markdown("""
                **Update Schedule**  
                Scraping: Hourly  
                Processing: On-demand  
                ML Models: Daily retraining
            """)
        
        with col3:
            st.markdown("""
                **System Version**  
                Dashboard: 1.0.0  
                Pipeline: 1.0.0  
                Last Updated: Today
            """)
        
        st.caption("""
            Disclaimer: This dashboard is for monitoring and analysis purposes only. 
            Always refer to official weather sources for critical weather information.
            Data provided by the National Oceanic and Atmospheric Administration (NOAA).
        """)
    
    def run(self):
        """Run the dashboard"""
        # Display header
        self.display_header()
        
        # Check data status
        if not self.display_data_status():
            return
        
        # Display metrics
        self.display_metrics()
        
        # Display main content
        self.display_main_charts()
        self.display_data_tables()
        
        # Display footer
        self.display_footer()

def main():
    """Main function to run the dashboard"""
    try:
        # Initialize dashboard
        dashboard = WeatherAnomalyDashboard()
        
        # Run dashboard
        dashboard.run()
        
    except Exception as e:
        st.error(f"Critical error in dashboard: {str(e)}")
        
        # Show recovery instructions
        st.markdown("""
        ### Recovery Steps:
        
        1. Check if all files exist:
        ```
        ls -la data/raw/ data/processed/ data/output/
        ```
        
        2. Run the pipeline again:
        ```
        python run_dashboard.py --pipeline
        ```
        
        3. Restart the dashboard:
        ```
        python run_dashboard.py --dashboard
        ```
        
        4. If problems persist, check logs:
        ```
        tail -f logs/*.log
        ```
        """)

if __name__ == "__main__":
    main()
