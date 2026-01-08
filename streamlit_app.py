#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point for Weather Anomaly Detection Dashboard
"""
import os
import sys
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

# Set page config first
st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create all required directories
def setup_project():
    """Setup the project structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/output',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create empty data files if they don't exist
    from config import Config
    
    # Create empty CSV files with headers if they don't exist
    sample_data = {
        'timestamp': [],
        'title': [],
        'description': [],
        'region': [],
        'alert_type': [],
        'severity': [],
        'source': []
    }
    
    if not os.path.exists(Config.RAW_DATA_PATH):
        pd.DataFrame(sample_data).to_csv(Config.RAW_DATA_PATH, index=False)
    
    if not os.path.exists(Config.PROCESSED_DATA_PATH):
        pd.DataFrame(sample_data).to_csv(Config.PROCESSED_DATA_PATH, index=False)
    
    if not os.path.exists(Config.ANOMALY_OUTPUT_PATH):
        pd.DataFrame({'date': [], 'total_alerts': [], 'is_anomaly': []}).to_csv(Config.ANOMALY_OUTPUT_PATH, index=False)
    
    if not os.path.exists(Config.FORECAST_OUTPUT_PATH):
        pd.DataFrame({'date': [], 'forecast': []}).to_csv(Config.FORECAST_OUTPUT_PATH, index=False)
    
    return True

# Try to setup project
try:
    setup_project()
except:
    pass  # Ignore errors in setup

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check if we can import the dashboard
def check_dependencies():
    """Check if all required modules can be imported"""
    try:
        # Try to import key modules
        import pandas
        import numpy
        import plotly
        import sklearn
        import xgboost
        import streamlit
        return True
    except ImportError as e:
        return False

# Main app logic
def main():
    """Main application function"""
    
    # Check dependencies first
    if not check_dependencies():
        st.error("Missing dependencies. Please install requirements.txt")
        st.code("pip install -r requirements.txt")
        return
    
    # Try to import and run the dashboard
    try:
        from src.dashboard.app import main as run_dashboard
        run_dashboard()
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        
        # Show setup instructions
        st.markdown("""
        ## Setup Instructions
        
        This app requires the complete project structure. Please ensure:
        
        1. **All project files are uploaded** including:
           - `src/` directory with all modules
           - `config.py` file
           - `requirements.txt` file
        
        2. **For local testing**, run:
        ```bash
        pip install -r requirements.txt
        python run_dashboard.py --pipeline
        python run_dashboard.py --dashboard
        ```
        
        3. **The project structure should be:**
        ```
        weather_anomaly_dashboard/
        ├── streamlit_app.py          (this file)
        ├── requirements.txt
        ├── config.py
        ├── run_dashboard.py
        ├── src/
        │   ├── __init__.py
        │   ├── scraping/
        │   ├── preprocessing/
        │   ├── ml/
        │   ├── utils/
        │   └── dashboard/
        ├── data/
        ├── models/
        └── logs/
        ```
        """)
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
