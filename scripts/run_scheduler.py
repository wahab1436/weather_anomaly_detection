#!/usr/bin/env python3
"""
Standalone scheduler script for running the system.
"""
import os
import sys
import time
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import WeatherAnomalySystem

def setup_logging():
    """Setup logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "scheduler.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main scheduler function."""
    logger = setup_logging()
    
    logger.info("Starting Weather Anomaly Detection Scheduler")
    logger.info(f"Current time: {datetime.now()}")
    
    # Initialize system
    try:
        system = WeatherAnomalySystem()
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return 1
    
    # Run initial pipeline
    logger.info("Running initial pipeline...")
    try:
        success = system.run_complete_pipeline()
        if success:
            logger.info("Initial pipeline completed successfully")
        else:
            logger.warning("Initial pipeline completed with some failures")
    except Exception as e:
        logger.error(f"Initial pipeline failed: {str(e)}")
    
    # Start scheduler
    logger.info("Starting scheduled jobs...")
    try:
        system.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
