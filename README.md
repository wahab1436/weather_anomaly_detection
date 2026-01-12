# Weather Anomaly Detection Dashboard

A **production-ready dashboard** for detecting, forecasting, and presenting unusual weather alerts using real-world unstructured text from official sources. Designed for senior-level implementation, the project integrates automated web scraping, natural language preprocessing, machine learning, and an interactive Streamlit dashboard.

---

## Table of Contents

- [Project Objective](#project-objective)  
- [Data Sources](#data-sources)  
- [Project Structure](#project-structure)  
- [Data Flow Overview](#data-flow-overview)  
- [Analysis Types](#analysis-types)  
- [Dashboard Features](#dashboard-features)  
- [Data Refresh & Performance](#data-refresh--performance)  
- [Senior-Level Features](#senior-level-features)  
- [Installation](#installation)  
- [Usage](#usage)  

---

## Project Objective

The goal of this project is to **detect and forecast unusual weather alerts** and present clear, plain-English insights for non-technical users.  

**Key Deliverables:**

- Hourly-updated dashboard for real-time monitoring  
- Detection of anomalies in weather alert spikes  
- Forecasting future alerts  
- Readable insights for a broad audience  

---

## Data Sources

- **Primary Source:** [Weather.gov Alerts](https://www.weather.gov/alerts)  
- **Supplementary Source:** [Forecast Discussions](https://www.weather.gov/wrh/TextProduct)  

**Type:** Real-world, unstructured textual weather alerts (not toy datasets)  

---


- **Modular Design:** Scraping → Preprocessing → ML → Dashboard  
- **Reusable Packages:** `__init__.py` included for clean imports  

---

## Data Flow Overview

1. **Web Scraping**  
   - Collect unstructured alerts hourly  
   - Save to `data/raw/weather_alerts_raw.csv`  

2. **Preprocessing**  
   - Clean text: lowercase, remove punctuation, stopwords  
   - Extract keywords and counts (TF-IDF optional)  
   - Aggregate by day, type, and region  
   - Save processed data for ML (`data/processed/`) and dashboard (`data/output/`)  

3. **Machine Learning**  
   - **Anomaly Detection:** Isolation Forest → flags unusual spikes  
   - **Forecasting:** XGBoost / LightGBM → predicts next-day/next-week alert counts  
   - Outputs saved in `data/output/` and models in `models/`  

4. **Dashboard (Streamlit)**  
   - Visualizations: line charts, bar charts, tables  
   - Filters: date, region, type  
   - Plain-English insights panel  
   - Cached data via `@st.cache_data` for performance  

---

## Analysis Types

- **Descriptive Analysis:** Alert counts, top keywords, type/region distribution  
- **Anomaly Analysis:** Spike detection in alert counts and keywords  
- **Correlation Analysis (Optional):** Relate weather metrics to alerts  
- **Predictive Analysis:** Forecast future alerts  
- **NLP/Text Analysis:** Trends and insights from unstructured text  

**Sample Insights:**

- “Flood alerts doubled compared to last week.”  
- “Storm warnings spiked in the northern region on Jan 5th.”  
- “Tomorrow, high probability of alerts in northern regions.”  

---

## Dashboard Features

- Hourly-updated alert monitoring  
- Line charts highlighting anomalies  
- Region- and type-wise breakdowns  
- Raw alert table snippets  
- Plain-English insights panel  
- Last updated timestamp displayed prominently  

---

## Data Refresh & Performance

- **Scraping Frequency:** Hourly  
- **Preprocessing & ML:** Runs on new raw data → outputs cached  
- **Dashboard:** Reads cached outputs → fast, production-ready  

---

## Features

- Modular, professional folder structure  
- Production-level caching and hourly refresh  
- Clean, minimal dashboard design  
- Plain-English insights for all users  
- Pre-trained ML models for fast anomaly detection and forecasting  
- End-to-end flow: Scraping → Preprocessing → ML → Dashboard → Insights  



