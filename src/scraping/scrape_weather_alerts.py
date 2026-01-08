"""
Web scraping module for collecting weather alerts from weather.gov
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config

# Configure logging - CREATE LOGS DIRECTORY FIRST
def setup_scraping_logger():
    """Setup logging for scraping module"""
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(f'{logs_dir}/scraping.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Setup logger
logger = setup_scraping_logger()

class WeatherAlertScraper:
    """Scrape weather alerts from official sources"""
    
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.alerts_url = Config.ALERTS_URL
        self.forecast_url = Config.FORECAST_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.retry_count = 0
        
    def make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        try:
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            self.retry_count = 0
            return response
        except requests.exceptions.RequestException as e:
            self.retry_count += 1
            if self.retry_count <= Config.MAX_RETRIES:
                logger.warning(f"Request failed (attempt {self.retry_count}): {e}")
                time.sleep(2 ** self.retry_count)  # Exponential backoff
                return self.make_request(url)
            else:
                logger.error(f"Max retries exceeded for {url}: {e}")
                return None
    
    def scrape_alerts(self) -> List[Dict]:
        """Scrape current weather alerts"""
        logger.info(f"Scraping alerts from {self.alerts_url}")
        
        response = self.make_request(self.alerts_url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            alerts = []
            
            # Look for alert containers - adapt to actual page structure
            # Common patterns in weather.gov alerts page
            alert_selectors = [
                'div.alert-item',
                'div.alertentry',
                'div.alert',
                'article.alert',
                'div[class*="alert"]'
            ]
            
            for selector in alert_selectors:
                alert_containers = soup.select(selector)
                if alert_containers:
                    logger.info(f"Found {len(alert_containers)} alerts with selector: {selector}")
                    for container in alert_containers:
                        alert_data = self._parse_alert_container(container)
                        if alert_data:
                            alerts.append(alert_data)
                    break
            
            # Fallback: look for any div with alert-like content
            if not alerts:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    text = div.get_text().lower()
                    if any(keyword in text for keyword in ['warning', 'watch', 'advisory', 'alert']):
                        alert_data = self._parse_generic_alert(div)
                        if alert_data:
                            alerts.append(alert_data)
            
            logger.info(f"Total alerts scraped: {len(alerts)}")
            return alerts
            
        except Exception as e:
            logger.error(f"Error parsing alerts: {e}")
            return []
    
    def _parse_alert_container(self, container) -> Optional[Dict]:
        """Parse individual alert container"""
        try:
            # Extract title
            title_elem = container.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
            title = title_elem.get_text(strip=True) if title_elem else "Weather Alert"
            
            # Extract description
            desc_elem = container.find(['p', 'div', 'span'])
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # If description is too short, get more text
            if len(description) < 20:
                description = container.get_text(strip=True)
                # Remove title from description if present
                if title in description:
                    description = description.replace(title, '').strip()
            
            # Extract location/region
            region = self._extract_region(title + " " + description)
            
            # Extract alert type
            alert_type = self._classify_alert_type(title, description)
            
            # Extract severity
            severity = self._extract_severity(title)
            
            # Timestamp
            timestamp = datetime.utcnow()
            
            # Generate unique ID
            alert_id = f"{timestamp.strftime('%Y%m%d%H%M%S')}_{hash(title) % 10000:04d}"
            
            return {
                'alert_id': alert_id,
                'timestamp': timestamp.isoformat(),
                'title': title[:500],  # Limit length
                'description': description[:2000],  # Limit length
                'region': region,
                'alert_type': alert_type,
                'severity': severity,
                'source': 'weather.gov',
                'scraped_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Parse error for alert container: {e}")
            return None
    
    def _parse_generic_alert(self, element) -> Optional[Dict]:
        """Parse generic alert element when specific structure not found"""
        try:
            text = element.get_text(strip=True)
            if len(text) < 50:  # Too short to be meaningful
                return None
            
            # Take first line as title
            lines = text.split('\n')
            title = lines[0][:200] if lines else "Weather Alert"
            description = text[:1500]
            
            region = self._extract_region(text)
            alert_type = self._classify_alert_type(title, description)
            severity = self._extract_severity(title)
            
            return {
                'alert_id': f"generic_{hash(text) % 1000000:06d}",
                'timestamp': datetime.utcnow().isoformat(),
                'title': title,
                'description': description,
                'region': region,
                'alert_type': alert_type,
                'severity': severity,
                'source': 'weather.gov',
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Generic parse error: {e}")
            return None
    
    def _extract_region(self, text: str) -> str:
        """Extract region from alert text"""
        text_lower = text.lower()
        
        for region in Config.REGIONS:
            if region in text_lower:
                return region.capitalize()
        
        # Try to extract state abbreviations
        states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
                  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
                  'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
                  'VA','WA','WV','WI','WY']
        
        for state in states:
            if f" {state} " in f" {text} ":
                return state
        
        return "National"
    
    def _classify_alert_type(self, title: str, description: str) -> str:
        """Classify alert type based on keywords"""
        text = f"{title} {description}".lower()
        
        for alert_type, keywords in Config.ALERT_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return alert_type
        
        return 'other'
    
    def _extract_severity(self, title: str) -> str:
        """Extract severity level from title"""
        title_lower = title.lower()
        
        for severity, keywords in Config.SEVERITY_KEYWORDS.items():
            if any(keyword in title_lower for keyword in keywords):
                return severity
        
        return 'unknown'
    
    def scrape_forecast_discussions(self) -> List[Dict]:
        """Scrape forecast discussions"""
        logger.info(f"Scraping forecast discussions from {self.forecast_url}")
        
        response = self.make_request(self.forecast_url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            discussions = []
            
            # Find forecast discussion links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                
                if 'discussion' in text or 'afd' in text or 'forecast discussion' in text:
                    full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                    
                    discussion_content = self._scrape_discussion_page(full_url)
                    if discussion_content:
                        discussions.append({
                            'alert_id': f"discussion_{hash(full_url) % 1000000:06d}",
                            'timestamp': datetime.utcnow().isoformat(),
                            'title': text[:200],
                            'description': discussion_content[:2000],
                            'region': 'National',
                            'alert_type': 'discussion',
                            'severity': 'information',
                            'source': 'weather.gov',
                            'scraped_at': datetime.utcnow().isoformat(),
                            'url': full_url
                        })
            
            logger.info(f"Total discussions scraped: {len(discussions)}")
            return discussions
            
        except Exception as e:
            logger.error(f"Error scraping forecast discussions: {e}")
            return []
    
    def _scrape_discussion_page(self, url: str) -> Optional[str]:
        """Scrape individual discussion page"""
        try:
            response = self.make_request(url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for pre tags (common for forecast discussions)
            pre_content = soup.find('pre')
            if pre_content:
                return pre_content.get_text(strip=True)[:5000]
            
            # Fallback to main content
            main_content = soup.find(['main', 'article', 'div.content'])
            if main_content:
                return main_content.get_text(strip=True)[:5000]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error scraping discussion page {url}: {e}")
            return None
    
    def merge_with_existing(self, new_data: List[Dict], existing_path: str) -> pd.DataFrame:
        """Merge new data with existing data, removing duplicates"""
        try:
            if os.path.exists(existing_path):
                existing_df = pd.read_csv(existing_path)
                logger.info(f"Loaded existing data: {len(existing_df)} rows")
            else:
                existing_df = pd.DataFrame()
            
            if new_data:
                new_df = pd.DataFrame(new_data)
                
                if not existing_df.empty:
                    # Remove duplicates based on title and timestamp
                    combined = pd.concat([existing_df, new_df])
                    combined = combined.drop_duplicates(
                        subset=['title', 'timestamp'], 
                        keep='last'
                    )
                    logger.info(f"Merged data: {len(combined)} total rows (+{len(new_df) - (len(combined) - len(existing_df))} new)")
                else:
                    combined = new_df
                    logger.info(f"Created new dataset: {len(combined)} rows")
                
                return combined
            else:
                logger.info("No new data to merge")
                return existing_df
                
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return pd.DataFrame(new_data) if new_data else pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved data to {filepath} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def run_scraping_job(self, raw_data_path: str = None) -> int:
        """Run complete scraping job"""
        if raw_data_path is None:
            raw_data_path = Config.RAW_DATA_PATH
        
        logger.info("=" * 60)
        logger.info("Starting weather alert scraping job")
        logger.info(f"Time: {datetime.utcnow().isoformat()}")
        logger.info("=" * 60)
        
        # Scrape alerts
        alerts = self.scrape_alerts()
        
        # Scrape forecast discussions
        discussions = self.scrape_forecast_discussions()
        
        # Combine all data
        all_data = alerts + discussions
        
        if all_data:
            # Merge with existing data
            merged_df = self.merge_with_existing(all_data, raw_data_path)
            
            # Save to CSV
            self.save_data(merged_df, raw_data_path)
            
            logger.info(f"Scraping job completed. Total records: {len(merged_df)}")
            return len(all_data)
        else:
            logger.warning("No data scraped in this run")
            return 0

def schedule_scraping():
    """Schedule the scraping job to run hourly"""
    from schedule import every, repeat, run_pending
    
    scraper = WeatherAlertScraper()
    
    @repeat(every(1).hour)
    def job():
        logger.info("Running scheduled scraping job")
        scraper.run_scraping_job()
        logger.info("Scheduled job completed")
    
    logger.info("Starting scraping scheduler (hourly)")
    while True:
        try:
            run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    # Create logs directory if not exists
    os.makedirs('logs', exist_ok=True)
    
    # Run once for immediate execution
    scraper = WeatherAlertScraper()
    count = scraper.run_scraping_job()
    print(f"Scraped {count} new alerts")
