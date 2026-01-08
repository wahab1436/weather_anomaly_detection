#!/usr/bin/env python3
"""
Setup script for Weather Anomaly Detection Dashboard
This script sets up the complete project structure and dependencies
"""
import os
import sys
import subprocess
from pathlib import Path
import logging

def create_project_structure():
    """Create the complete project folder structure"""
    base_dirs = [
        "data/raw",
        "data/processed", 
        "data/output",
        "models",
        "logs",
        "notebooks",
        "src/scraping",
        "src/preprocessing",
        "src/ml",
        "src/utils",
        "src/dashboard"
    ]
    
    for directory in base_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    return True

def create_init_files():
    """Create all __init__.py files"""
    init_files = [
        "src/__init__.py",
        "src/scraping/__init__.py",
        "src/preprocessing/__init__.py",
        "src/ml/__init__.py",
        "src/utils/__init__.py",
        "src/dashboard/__init__.py"
    ]
    
    # Create basic __init__.py content
    init_content = """\"\"\"Module initialization\"\"\"\n\n__version__ = "1.0.0"\n"""
    
    for file_path in init_files:
        with open(file_path, 'w') as f:
            f.write(init_content)
        print(f"Created: {file_path}")
    
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies from requirements.txt...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("requirements.txt not found. Creating default...")
            create_requirements_file()
        
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements = """streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
beautifulsoup4==4.12.2
requests==2.31.0
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.0
nltk==3.8.1
joblib==1.3.2
schedule==1.2.0
python-dotenv==1.0.0
scipy==1.11.4
statsmodels==0.14.0
pyyaml==6.0.1
tqdm==4.66.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("Created requirements.txt")
    return True

def create_config_file():
    """Create configuration file"""
    config_content = """{
    "scraping": {
        "base_url": "https://www.weather.gov",
        "user_agent": "WeatherAnomalyDetection/1.0",
        "interval_seconds": 3600,
        "max_retries": 3,
        "timeout": 30
    },
    "processing": {
        "min_text_length": 10,
        "max_text_length": 10000,
        "stopwords_language": "english",
        "vectorizer_max_features": 1000
    },
    "ml": {
        "anomaly_contamination": 0.05,
        "forecast_horizon": 7,
        "test_size": 0.2,
        "random_state": 42,
        "xgboost_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8
        }
    },
    "dashboard": {
        "port": 8501,
        "host": "0.0.0.0",
        "theme": "light",
        "cache_ttl": 300,
        "max_display_rows": 1000
    },
    "paths": {
        "raw_data": "data/raw/weather_alerts_raw.csv",
        "processed_data": "data/processed/weather_alerts_processed.csv",
        "aggregated_data": "data/processed/weather_alerts_aggregated.csv",
        "anomaly_output": "data/output/anomaly_results.csv",
        "forecast_output": "data/output/forecast_results.csv",
        "anomaly_model": "models/isolation_forest.pkl",
        "forecast_model": "models/xgboost_forecast.pkl"
    }
}
"""
    
    with open("config.json", "w") as f:
        f.write(config_content)
    print("Created config.json")
    return True

def create_sample_notebooks():
    """Create sample Jupyter notebooks"""
    
    # EDA notebook
    eda_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Weather Alert EDA\n",
    "## Exploratory Data Analysis for Weather Anomaly Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "\n",
    "# Set style\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette(\"husl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "df = pd.read_csv('../data/processed/weather_alerts_processed.csv')\n",
    "print(f\"Data shape: {df.shape}\")\n",
    "print(f\"Columns: {df.columns.tolist()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Statistics"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open("notebooks/EDA.ipynb", "w") as f:
        f.write(eda_content)
    
    # NLP notebook
    nlp_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NLP Feature Extraction\n",
    "## Text Analysis for Weather Alerts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer\n",
    "import matplotlib.pyplot as plt\n",
    "from wordcloud import WordCloud\n",
    "\n",
    "# Download NLTK data\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open("notebooks/NLP_feature_extraction.ipynb", "w") as f:
        f.write(nlp_content)
    
    print("Created sample notebooks")
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Data files
data/
models/
*.csv
*.pkl
*.joblib
*.parquet
*.feather

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Virtual environment
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Streamlit
.streamlit/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Created .gitignore")
    return True

def create_readme():
    """Create README.md file"""
    readme_content = """# Weather Anomaly Detection Dashboard

## Overview
A production-ready system for detecting and forecasting unusual weather patterns using unstructured text data from official sources.

## Features
- **Real-time Monitoring**: Hourly-updated dashboard
- **Anomaly Detection**: Identifies unusual spikes in alert patterns
- **Forecasting**: Predicts future alert counts
- **Plain-English Insights**: Clear, readable insights for non-technical users

## Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd weather_anomaly_dashboard
python setup.py
