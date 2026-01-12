"""
Scheduler script for periodic pipeline execution
"""

import time
import logging
from datetime import datetime

from main import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Scheduler started")

    while True:
        try:
            logger.info(f"Running pipeline at {datetime.utcnow()}")
            run_pipeline()
            logger.info("Pipeline completed")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

        # Run once every 6 hours
        time.sleep(6 * 60 * 60)


if __name__ == "__main__":
    main()
