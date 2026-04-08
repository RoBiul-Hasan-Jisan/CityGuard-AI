#  CityGuard - Crime Intelligence Dashboard

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.0+-red.svg)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

> **Advanced Crime Analytics Platform for San Francisco Police Department**
>
> Real-time crime intelligence, anomaly detection, and predictive analytics for data-driven policing

## Live Demo Preview

![CityGuard Dashboard Preview](https://github.com/RoBiul-Hasan-Jisan/CityGuard-AI/blob/main/image.png)

- [▶️ Watch Demo](https://github.com/RoBiul-Hasan-Jisan/CityGuard-AI/blob/main/demo.mp4)
  
- **Kaggle:** [robiulhasanjisan](https://www.kaggle.com/code/robiulhasanjisan/crime-analysis)

##  Key Features

###  **Real-time Analytics**
- **Weekly Crime Trends** with anomaly detection (Z-score based)
- **Week-over-Week Percentage Changes** with spike identification
- **Rolling Averages** (4-week, 8-week, 12-week)
- **Automated Alert System** for unusual patterns

###  **Geospatial Intelligence**
- **Interactive Folium Maps** with district overlays
- **Heatmap Visualization** for crime density
- **District Ranking System** (1-10 scale)
- **Geographical Clustering** (K-Means algorithm)

###  **Pattern Recognition**
- **Temporal Analysis** (Hour, Day, Month, Year patterns)
- **Category Distribution** with top crime types
- **Day × Hour Heatmaps** for precise timing
- **Year-over-Year Trend Analysis**

###  **Machine Learning**
- **K-Means Clustering** for hotspot detection
- **Anomaly Detection** using statistical methods
- **Predictive Feature Engineering**
- **Cluster-based Resource Allocation**

###  **Data Export**
- **CSV Export** for all datasets
- **Filtered Data Export** based on user selections
- **Weekly Summary Reports**
- **District Rankings Export**

##  Quick Start

### Prerequisites
**Installation**
Clone the repository

```bash
git clone https://github.com/yourusername/cityguard.git
cd cityguard
```
---
**Install dependencies**

```bash
pip install -r requirements.txt
Download the dataset
```
---

```bash
# Download from Kaggle: SF Crime Dataset
# Place train.csv in the data/ directory
mkdir data
# Move train.csv to data/train.csv
```
---
**Run the application**

- Streamlit Version:

``` bash
streamlit run app_streamlit.py
```
---
- Flask Version:

```bash
python app_flask.py
```
---

**Open your browser**

```bash
http://localhost:5000 (Flask)
http://localhost:8501 (Streamlit)
```
---

```bash
Python 3.8 or higher
pip package manager
```
---

## Project Structure
``` bash
cityguard/
├── app_flask.py              
├── app_streamlit.py          # Streamlit application
├── requirements.txt          
├── README.md                
├── data/
│   └── train.csv         
├── templates/               
│   └── index.html
├── outputs/                 
│   ├── visualizations/
│   ├── sf_crime_map.html
│   └── crime_insights_panel.html
└── notebooks/               
    └── analysis.ipynb
```
---

### Dependencies
```txt
# Core
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
folium>=0.12.0

# Machine Learning
scikit-learn>=0.24.0

# Web Framework
streamlit>=1.0.0
flask>=2.0.0
flask-caching>=1.10.0

# Utilities
streamlit-folium>=0.6.0
```
---
## Use Cases

- **For Law Enforcement**
  
  - Resource Allocation: Identify high-crime areas for patrol deployment

  - Predictive Policing: Forecast crime patterns based on historical data

  - Performance Metrics: Track district-level crime reduction

  - Emergency Response: Optimize response times using hotspot analysis

- **For Policy Makers**
  
  - Strategic Planning: Data-driven policy decisions

  - Budget Allocation: Evidence-based resource distribution

  - Public Safety Reports: Generate comprehensive crime reports

  - Community Outreach: Identify areas needing community programs

- **For Researchers**
  
  - Crime Pattern Analysis: Study temporal and spatial patterns

  - Anomaly Detection: Identify unusual crime events

  - Cluster Analysis: Understand crime concentration areas

  - Predictive Modeling: Develop new prediction algorithms

## Analytics Dashboard Components
```bash
1. Weekly Trends Tab
Line chart with anomaly highlighting

Percentage change bar chart

Spike detection table

Rolling average overlays

2. Geographic Analysis Tab
Interactive crime map with district circles

Heatmap overlay option

District ranking table

Crime category web visualization

3. Crime Patterns Tab
Top categories bar chart

Day-of-week distribution

Hourly pattern analysis

Day × Hour heatmap

Yearly trend analysis

4. Cluster Analysis Tab
K-Means clustering visualization

Adjustable number of clusters

Cluster statistics table

Centroid coordinates

Dominant crime per cluster

5. Insights Tab
Key statistics dashboard

Critical findings panel

Trend analysis gauge

Actionable recommendations

Data export options
```
---
## Acknowledgments
- SFPD for providing the crime incident data

- Kaggle for hosting the dataset

- Open Source Community for amazing libraries



##  Roadmap
- **Version 1.0 (Current)**
   - Basic analytics dashboard

   - Weekly trend analysis

   - Geographic visualization

   - Anomaly detection

- **Version 2.0 (Planned)**
   - Real-time data streaming

  - Predictive modeling (LSTM/Prophet)

  - Mobile application

  - Automated reporting system

- **Version 3.0 (Future)**
   -  AI-powered recommendations

  - Integration with 911 dispatch

  -  Cross-city comparison

  -  Social media sentiment analysis

## 👨‍💻 Author
**Robiul Hasan Jisan**

- **Portfolio:** [robiulhasanjisan.vercel.app](https://robiulhasanjisan.vercel.app/)
- **GitHub:** [@RoBiul-Hasan-Jisan](https://github.com/RoBiul-Hasan-Jisan)
- **Kaggle:** [robiulhasanjisan](https://www.kaggle.com/robiulhasanjisan)

