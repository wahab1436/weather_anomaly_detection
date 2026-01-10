"""
Web scraper for Weather.gov alerts with proper rate limiting and error handling.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional
import logging
from urllib.parse import urljoin
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherAlertScraper:
    """Professional weather alert scraper with rate limiting and error handling."""
    
    def __init__(self, base_url: str = "https://www.weather.gov"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherResearchBot/1.0 (Contact: research@example.com)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })
        self.request_delay = 2  # seconds between requests
        self.max_retries = 3
        
    def scrape_alerts_page(self, url: str) -> Optional[BeautifulSoup]:
        """Safely scrape a page with retry logic."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue
                    
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
                    return None
                    
        return None
    
    def parse_alert_entry(self, alert_div) -> Optional[Dict]:
        """Parse individual alert entry from the alerts page."""
        try:
            if not alert_div:
                return None
                
            title_elem = alert_div.find('a', class_='alert-title')
            if not title_elem:
                return None
                
            alert_title = title_elem.text.strip()
            alert_link = urljoin(self.base_url, title_elem.get('href', ''))
            
            details = {}
            rows = alert_div.find_all('div', class_='row')
            
            for row in rows:
                cols = row.find_all('div')
                if len(cols) >= 2:
                    key = cols[0].text.strip().replace(':', '').lower()
                    value = cols[1].text.strip()
                    details[key] = value
            
            alert_text_div = alert_div.find('div', class_='alert-text')
            alert_text = alert_text_div.text.strip() if alert_text_div else ""
            
            issued_date = None
            if 'issued' in details:
                try:
                    issued_date = pd.to_datetime(details['issued'])
                except:
                    issued_date = datetime.now(pytz.UTC)
            else:
                issued_date = datetime.now(pytz.UTC)
            
            alert_type = self._classify_alert_type(alert_title)
            region = details.get('area', 'Unknown')
            
            return {
                'alert_id': f"{int(issued_date.timestamp())}_{hash(alert_title) % 1000000}",
                'title': alert_title,
                'text': alert_text,
                'type': alert_type,
                'region': region,
                'issued_date': issued_date,
                'effective_date': details.get('effective', ''),
                'expires_date': details.get('expires', ''),
                'severity': details.get('severity', 'Unknown'),
                'certainty': details.get('certainty', 'Unknown'),
                'urgency': details.get('urgency', 'Unknown'),
                'link': alert_link,
                'scraped_at': datetime.now(pytz.UTC)
            }
            
        except Exception as e:
            logger.error(f"Error parsing alert entry: {str(e)}")
            return None
    
    def _classify_alert_type(self, title: str) -> str:
        """Classify alert type based on title keywords."""
        if not title:
            return 'other'
            
        title_lower = title.lower()
        
        alert_types = {
            'flood': ['flood', 'flooding', 'flash flood', 'river flood'],
            'storm': ['storm', 'thunderstorm', 'severe storm', 'hail', 'tornado'],
            'wind': ['wind', 'high wind', 'wind advisory'],
            'winter': ['winter', 'snow', 'ice', 'blizzard', 'freez', 'frost'],
            'fire': ['fire', 'red flag', 'fire weather'],
            'heat': ['heat', 'excessive heat', 'heat advisory'],
            'cold': ['cold', 'wind chill', 'extreme cold'],
            'coastal': ['coastal', 'surf', 'high surf', 'rip current'],
            'air': ['air quality', 'air stagnation'],
            'other': ['special weather', 'advisory', 'watch', 'warning']
        }
        
        for alert_type, keywords in alert_types.items():
            if any(keyword in title_lower for keyword in keywords):
                return alert_type
                
        return 'other'
    
    def scrape_all_alerts(self) -> List[Dict]:
        """Scrape all active weather alerts."""
        alerts_url = f"{self.base_url}/alerts"
        soup = self.scrape_alerts_page(alerts_url)
        
        if not soup:
            logger.warning("Failed to scrape alerts page")
            return []  # Return empty list, not None
        
        alert_containers = soup.find_all('div', class_='alert-item')
        
        # FIX: Check if alert_containers is not None
        if alert_containers is None:
            alert_containers = []
            
        logger.info(f"Found {len(alert_containers)} alert containers")
        
        alerts = []
        
        for container in alert_containers:
            alert_data = self.parse_alert_entry(container)
            if alert_data:
                alerts.append(alert_data)
        
        if not alerts:
            alerts = self._fallback_parsing(soup)
        
        logger.info(f"Successfully parsed {len(alerts)} alerts")
        return alerts  # Always returns a list (empty or with items)
    
    def _fallback_parsing(self, soup: BeautifulSoup) -> List[Dict]:
        """Fallback parsing method if structured parsing fails."""
        alerts = []
        
        if not soup:
            return alerts
            
        alert_sections = soup.find_all(['div', 'section'], class_=lambda x: x and 'alert' in x.lower())
        
        for section in alert_sections:
            text = section.get_text(strip=True, separator=' ')
            if text and len(text) > 50:
                alert_data = {
                    'alert_id': f"fallback_{hash(text) % 1000000}",
                    'title': 'Weather Alert',
                    'text': text[:1000],
                    'type': 'other',
                    'region': 'Unknown',
                    'issued_date': datetime.now(pytz.UTC),
                    'effective_date': '',
                    'expires_date': '',
                    'severity': 'Unknown',
                    'certainty': 'Unknown',
                    'urgency': 'Unknown',
                    'link': self.base_url + '/alerts',
                    'scraped_at': datetime.now(pytz.UTC)
                }
                alerts.append(alert_data)
        
        return alerts
    
    def save_alerts_to_csv(self, alerts: List[Dict], filepath: str) -> int:
        """Save alerts to CSV and return count saved."""
        if not alerts:
            logger.warning("No alerts to save")
            return 0  # Return 0, not None
        
        try:
            df = pd.DataFrame(alerts)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if os.path.exists(filepath):
                try:
                    existing_df = pd.read_csv(filepath)
                    combined_df = pd.concat([existing_df, df]).drop_duplicates(
                        subset=['alert_id'], 
                        keep='last'
                    )
                    combined_df.to_csv(filepath, index=False)
                    new_count = len(df) - len(existing_df)
                    # Ensure non-negative count
                    new_count = max(new_count, 0)
                    logger.info(f"Appended {new_count} new alerts to {filepath}")
                    return int(new_count)  # Ensure integer
                except Exception as e:
                    logger.error(f"Error appending to existing file: {str(e)}")
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {len(df)} alerts to {filepath}")
                    return int(len(df))  # Ensure integer
            else:
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} alerts to {filepath}")
                return int(len(df))  # Ensure integer
                
        except Exception as e:
            logger.error(f"Error saving alerts to CSV: {str(e)}")
            return 0  # Return 0 on error

def main() -> int:
    """Main scraping function. Returns number of alerts scraped."""
    scraper = WeatherAlertScraper()
    alert_count = 0  # Initialize to 0
    
    try:
        alerts = scraper.scrape_all_alerts()
        
        if alerts:
            timestamp = datetime.now().strftime('%Y%m%d')
            filepath = f"data/raw/weather_alerts_{timestamp}.csv"
            
            # Save to timestamped file
            count1 = scraper.save_alerts_to_csv(alerts, filepath)
            
            # Save to main raw data file
            main_filepath = "data/raw/weather_alerts_raw.csv"
            count2 = scraper.save_alerts_to_csv(alerts, main_filepath)
            
            # Use the maximum count or length of alerts
            alert_count = max(count1, count2, len(alerts))
            alert_count = int(alert_count)  # Ensure integer
            logger.info(f"Scraping completed successfully. Processed {alert_count} alerts.")
        else:
            logger.warning("No alerts were scraped.")
            alert_count = 0  # Explicitly set to 0
            
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        alert_count = 0  # Return 0 on failure
    
    # ALWAYS return an integer
    return int(alert_count)

if __name__ == "__main__":
    result = main()
    print(f"Scraping completed. Alerts processed: {result}")
    sys.exit(0 if result > 0 else 1)
