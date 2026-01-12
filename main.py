#!/usr/bin/env python3

"""
Main entry point for Weather Anomaly Detection System
"""

import os
import sys
from pathlib import Path

# ---------------- PATH SETUP ----------------
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------- IMPORTS ----------------
from scripts.initial_data_collection import run_initial_collection
from src.preprocessing.preprocess_text import preprocess_pipeline
from src.dashboard.app import run_pipeline, run_dashboard

# Optional ML imports
try:
    from src.ml.anomaly_detection import run_anomaly_detection
    from src.ml.forecast_model import run_forecasting
except Exception:
    run_anomaly_detection = None
    run_forecasting = None


# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print(" WEATHER ANOMALY DETECTION SYSTEM ")
    print("=" * 60)

    print("\nSelect option:")
    print("1. Run Initial Data Collection")
    print("2. Run Full Pipeline")
    print("3. Launch Dashboard")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        run_initial_collection(days_back=7)

    elif choice == "2":
        run_pipeline()

    elif choice == "3":
        run_dashboard()

    elif choice == "4":
        print("Exiting...")
        return

    else:
        print("Invalid choice")

    input("\nPress Enter to continue...")
    main()


if __name__ == "__main__":
    main()
