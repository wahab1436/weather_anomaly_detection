import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherAlertScraper:
    """Scrapes weather alerts from official sources with respect for rate limits."""
    
    def __init__(self, base_url: str = "https://www.weather.gov"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherResearchBot/1.0 (Contact: research@example.com)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.rate_limit_delay = 2  # seconds between requests
        
    def scrape_alerts(self) -> pd.DataFrame:
        """
        Scrape current weather alerts from weather.gov/alerts
        
        Returns:
            DataFrame with alert information or empty DataFrame if scraping fails
        """
        try:
            alerts_url = f"{self.base_url}/alerts"
            logger.info(f"Scraping alerts from {alerts_url}")
            
            response = self.session.get(alerts_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find alert data - structure may vary, this is a robust approach
            alerts_data = []
            
            # Look for alert containers (common patterns on weather.gov)
            alert_elements = soup.find_all(['div', 'article'], class_=lambda x: x and 'alert' in x.lower())
            
            if not alert_elements:
                # Alternative: look for CAP (Common Alerting Protocol) data
                cap_elements = soup.find_all('entry')  # Atom feed entries
                if cap_elements:
                    alerts_data = self._parse_cap_alerts(cap_elements)
                else:
                    # Fallback to table parsing
                    tables = soup.find_all('table')
                    for table in tables:
                        alerts_data.extend(self._parse_table_alerts(table))
            else:
                alerts_data = self._parse_alert_elements(alert_elements)
            
            # If no alerts found, return empty with proper structure
            if not alerts_data:
                logger.warning("No alerts found in the page")
                return pd.DataFrame(columns=[
                    'alert_id', 'title', 'description', 'region', 'alert_type',
                    'severity', 'effective', 'expires', 'issued', 'scraped_at'
                ])
            
            df = pd.DataFrame(alerts_data)
            df['scraped_at'] = datetime.now()
            
            # Remove exact duplicates based on alert_id and issued time
            if 'alert_id' in df.columns:
                df = df.drop_duplicates(subset=['alert_id', 'issued'], keep='first')
            
            logger.info(f"Successfully scraped {len(df)} alerts")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Network error during scraping: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error during scraping: {str(e)}")
            return pd.DataFrame()
    
    def _parse_alert_elements(self, alert_elements: List) -> List[Dict]:
        """Parse alert HTML elements."""
        alerts = []
        for element in alert_elements:
            try:
                alert = {}
                
                # Extract title
                title_elem = element.find(['h2', 'h3', 'h4', 'strong', 'b'])
                alert['title'] = title_elem.get_text(strip=True) if title_elem else "No Title"
                
                # Extract description
                desc_elem = element.find(['p', 'div'], class_=lambda x: x and 'desc' in str(x).lower())
                alert['description'] = desc_elem.get_text(strip=True) if desc_elem else ""
                
                # Extract region from title or description
                alert['region'] = self._extract_region(alert['title'] + " " + alert['description'])
                
                # Extract alert type
                alert['alert_type'] = self._extract_alert_type(alert['title'])
                
                # Extract severity
                alert['severity'] = self._extract_severity(alert['title'] + " " + alert['description'])
                
                # Generate unique ID
                alert['alert_id'] = f"{alert['alert_type']}_{alert['region']}_{int(time.time())}"
                
                # Timestamps
                now = datetime.now()
                alert['effective'] = now
                alert['expires'] = now + timedelta(hours=6)  # Default 6 hours
                alert['issued'] = now
                
                alerts.append(alert)
                
            except Exception as e:
                logger.warning(f"Failed to parse alert element: {str(e)}")
                continue
        
        return alerts
    
    def _parse_cap_alerts(self, cap_elements: List) -> List[Dict]:
        """Parse CAP format alerts."""
        alerts = []
        for entry in cap_elements:
            try:
                alert = {}
                alert['title'] = entry.find('title').text if entry.find('title') else "No Title"
                alert['description'] = entry.find('summary').text if entry.find('summary') else ""
                alert['region'] = self._extract_region(alert['title'])
                alert['alert_type'] = self._extract_alert_type(alert['title'])
                alert['severity'] = self._extract_severity(alert['title'])
                alert['alert_id'] = entry.find('id').text if entry.find('id') else f"cap_{int(time.time())}"
                
                # Parse timestamps if available
                now = datetime.now()
                alert['effective'] = now
                alert['expires'] = now + timedelta(hours=6)
                alert['issued'] = now
                
                alerts.append(alert)
            except Exception as e:
                logger.warning(f"Failed to parse CAP alert: {str(e)}")
                continue
        
        return alerts
    
    def _parse_table_alerts(self, table) -> List[Dict]:
        """Parse table-based alert data."""
        alerts = []
        try:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all('td')
                if len(cells) >= 3:
                    alert = {
                        'title': cells[0].get_text(strip=True),
                        'region': cells[1].get_text(strip=True),
                        'alert_type': cells[2].get_text(strip=True),
                        'description': cells[3].get_text(strip=True) if len(cells) > 3 else "",
                        'severity': 'Unknown',
                        'alert_id': f"table_{int(time.time())}",
                        'effective': datetime.now(),
                        'expires': datetime.now() + timedelta(hours=6),
                        'issued': datetime.now()
                    }
                    alerts.append(alert)
        except Exception as e:
            logger.warning(f"Failed to parse table: {str(e)}")
        
        return alerts
    
    def _extract_region(self, text: str) -> str:
        """Extract region from text."""
        text = text.lower()
        regions = [
            'northeast', 'southeast', 'midwest', 'south', 'southwest',
            'west', 'northwest', 'central', 'eastern', 'western',
            'northern', 'southern'
        ]
        
        for region in regions:
            if region in text:
                return region.capitalize()
        
        # Fallback: extract state abbreviations
        states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID',
                 'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
                 'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
                 'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
                 'WI','WY']
        
        for state in states:
            if f" {state} " in f" {text} ":
                return state
        
        return "Unknown"
    
    def _extract_alert_type(self, text: str) -> str:
        """Extract alert type from text."""
        text = text.lower()
        alert_types = {
            'flood': 'Flood',
            'storm': 'Storm',
            'tornado': 'Tornado',
            'hurricane': 'Hurricane',
            'blizzard': 'Blizzard',
            'wind': 'Wind',
            'rain': 'Rain',
            'snow': 'Snow',
            'ice': 'Ice',
            'fog': 'Fog',
            'heat': 'Heat',
            'cold': 'Cold',
            'fire': 'Fire',
            'air': 'Air Quality',
            'coastal': 'Coastal',
            'winter': 'Winter Weather',
            'thunderstorm': 'Thunderstorm',
            'tsunami': 'Tsunami'
        }
        
        for key, value in alert_types.items():
            if key in text:
                return value
        
        return "Other"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity from text."""
        text = text.lower()
        severities = {
            'warning': 'Warning',
            'watch': 'Watch',
            'advisory': 'Advisory',
            'emergency': 'Emergency',
            'severe': 'Severe',
            'extreme': 'Extreme'
        }
        
        for key, value in severities.items():
            if key in text:
                return value
        
        return "Unknown"
    
    def scrape_forecast_discussions(self) -> Optional[str]:
        """Scrape forecast discussions text."""
        try:
            url = f"{self.base_url}/wrh/TextProduct"
            logger.info(f"Scraping forecast discussions from {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for forecast discussion text
            forecast_text = ""
            pre_elements = soup.find_all('pre')
            for pre in pre_elements:
                text = pre.get_text(strip=True)
                if 'discussion' in text.lower() or 'forecast' in text.lower():
                    forecast_text = text
                    break
            
            if not forecast_text:
                # Fallback: get all text from main content
                main = soup.find('main') or soup.find('body')
                forecast_text = main.get_text(strip=True) if main else ""
            
            logger.info(f"Scraped forecast discussion (length: {len(forecast_text)})")
            return forecast_text[:10000]  # Limit length
            
        except Exception as e:
            logger.error(f"Failed to scrape forecast discussions: {str(e)}")
            return None
    
    def save_alerts(self, df: pd.DataFrame, filepath: str):
        """Save alerts to CSV, appending new data only."""
        try:
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Remove duplicates based on key columns
                duplicate_cols = ['alert_id', 'issued'] if 'alert_id' in combined_df.columns else ['title', 'issued']
                combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep='last')
                
                combined_df.to_csv(filepath, index=False)
                logger.info(f"Appended {len(df)} new alerts to {filepath}")
            else:
                df.to_csv(filepath, index=False)
                logger.info(f"Created new file with {len(df)} alerts: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save alerts: {str(e)}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)

def main():
    """Main scraping function to be called by scheduler."""
    scraper = WeatherAlertScraper()
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    
    # Scrape alerts
    alerts_df = scraper.scrape_alerts()
    
    if not alerts_df.empty:
        # Save raw alerts
        raw_filepath = 'data/raw/weather_alerts_raw.csv'
        scraper.save_alerts(alerts_df, raw_filepath)
        
        # Also save as timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        backup_path = f'data/raw/backups/weather_alerts_{timestamp}.csv'
        os.makedirs('data/raw/backups', exist_ok=True)
        alerts_df.to_csv(backup_path, index=False)
        
        # Scrape forecast discussions
        forecast_text = scraper.scrape_forecast_discussions()
        if forecast_text:
            forecast_data = {
                'timestamp': datetime.now().isoformat(),
                'text': forecast_text
            }
            forecast_path = 'data/raw/forecast_discussions.json'
            if os.path.exists(forecast_path):
                with open(forecast_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data.append(forecast_data)
                with open(forecast_path, 'w') as f:
                    json.dump(existing_data, f)
            else:
                with open(forecast_path, 'w') as f:
                    json.dump([forecast_data], f)
        
        logger.info("Scraping completed successfully")
    else:
        logger.warning("No alerts scraped - data may be empty")

if __name__ == "__main__":
    main()
