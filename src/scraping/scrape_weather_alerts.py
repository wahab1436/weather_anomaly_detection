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
                
                # Check if we're being rate limited
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
            # Extract alert title and link
            title_elem = alert_div.find('a', class_='alert-title')
            if not title_elem:
                return None
                
            alert_title = title_elem.text.strip()
            alert_link = urljoin(self.base_url, title_elem.get('href', ''))
            
            # Extract alert details
            details = {}
            rows = alert_div.find_all('div', class_='row')
            
            for row in rows:
                cols = row.find_all('div')
                if len(cols) >= 2:
                    key = cols[0].text.strip().replace(':', '').lower()
                    value = cols[1].text.strip()
                    details[key] = value
            
            # Extract alert text
            alert_text_div = alert_div.find('div', class_='alert-text')
            alert_text = alert_text_div.text.strip() if alert_text_div else ""
            
            # Parse date
            issued_date = None
            if 'issued' in details:
                try:
                    issued_date = pd.to_datetime(details['issued'])
                except:
                    issued_date = datetime.now(pytz.UTC)
            else:
                issued_date = datetime.now(pytz.UTC)
            
            # Determine alert type from title
            alert_type = self._classify_alert_type(alert_title)
            
            # Extract region/state
            region = details.get('area', 'Unknown')
            
            return {
                'alert_id': f"{issued_date.timestamp()}_{hash(alert_title) % 1000000}",
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
            return []
        
        # Find alert containers
        alert_containers = soup.find_all('div', class_='alert-item')
        alerts = []
        
        logger.info(f"Found {len(alert_containers)} alert containers")
        
        for container in alert_containers:
            alert_data = self.parse_alert_entry(container)
            if alert_data:
                alerts.append(alert_data)
        
        # If no structured alerts found, try alternative parsing
        if not alerts:
            alerts = self._fallback_parsing(soup)
        
        logger.info(f"Successfully parsed {len(alerts)} alerts")
        return alerts
    
    def _fallback_parsing(self, soup: BeautifulSoup) -> List[Dict]:
        """Fallback parsing method if structured parsing fails."""
        alerts = []
        
        # Look for any alert-like content
        alert_sections = soup.find_all(['div', 'section'], class_=lambda x: x and 'alert' in x.lower())
        
        for section in alert_sections:
            # Try to extract text
            text = section.get_text(strip=True, separator=' ')
            if len(text) > 50:  # Reasonable minimum length for an alert
                alert_data = {
                    'alert_id': f"fallback_{hash(text) % 1000000}",
                    'title': 'Weather Alert',
                    'text': text[:1000],  # Limit text length
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
    
    def scrape_forecast_discussions(self) -> List[Dict]:
        """Scrape forecast discussions from WRH text products."""
        discussions_url = "https://www.weather.gov/wrh/TextProduct"
        soup = self.scrape_alerts_page(discussions_url)
        
        if not soup:
            return []
        
        discussions = []
        # Implementation depends on the actual structure of the page
        # This is a placeholder - you'll need to inspect the actual page structure
        
        return discussions
    
    def save_alerts_to_csv(self, alerts: List[Dict], filepath: str):
        """Save alerts to CSV with proper handling of existing data."""
        if not alerts:
            logger.warning("No alerts to save")
            return
        
        df = pd.DataFrame(alerts)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Append to existing file if it exists
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            # Remove duplicates based on alert_id
            combined_df = pd.concat([existing_df, df]).drop_duplicates(
                subset=['alert_id'], 
                keep='last'
            )
            combined_df.to_csv(filepath, index=False)
            logger.info(f"Appended {len(df)} new alerts to {filepath}")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} alerts to {filepath}")

def main():
    """Main scraping function to be called by scheduler."""
    scraper = WeatherAlertScraper()
    
    try:
        # Scrape alerts
        alerts = scraper.scrape_all_alerts()
        
        if alerts:
            # Define file path
            timestamp = datetime.now().strftime('%Y%m%d')
            filepath = f"data/raw/weather_alerts_{timestamp}.csv"
            
            # Save alerts
            scraper.save_alerts_to_csv(alerts, filepath)
            
            # Also save to main raw file for dashboard
            main_filepath = "data/raw/weather_alerts_raw.csv"
            scraper.save_alerts_to_csv(alerts, main_filepath)
            
            logger.info(f"Scraping completed successfully. Saved {len(alerts)} alerts.")
        else:
            logger.warning("No alerts were scraped.")
            
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
