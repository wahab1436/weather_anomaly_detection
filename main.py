#!/usr/bin/env python3
"""
Weather Anomaly Detection System - Entry Point
Simply launches the main system from src/dashboard/app.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main system
if __name__ == "__main__":
    try:
        from dashboard.app import main
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure src/dashboard/app.py exists")
