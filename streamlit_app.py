#!/usr/bin/env python3
"""
Streamlit app entry point for Weather Anomaly Detection Dashboard
"""
import os
import sys
from pathlib import Path

# Create required directories
directories = ['data/raw', 'data/processed', 'data/output', 'models', 'logs']
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the dashboard
from src.dashboard.app import main

if __name__ == "__main__":
    main()
