"""
CityGuard — Crime Intelligence Dashboard

"""

from flask import Flask, render_template, request, jsonify, send_file, abort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
import io
import base64
from datetime import datetime, timedelta
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from flask_caching import Cache

warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cityguard-secret-key-2024'
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Configure matplotlib for backend
plt.switch_backend('Agg')

# Set style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#cdd9e5",
    "grid.color": "#21262d",
})

# Global variables to store data
df_raw = None
df_filtered = None
weekly_data = None
districts_data = None


# DATA LOADING FUNCTIONS


def create_sample_data():
    """Create sample data for demonstration when real data isn't available"""
    np.random.seed(42)
    n_samples = 10000
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=n_samples)
    districts = ['MISSION', 'SOUTHERN', 'NORTHERN', 'BAYVIEW', 'RICHMOND', 
                 'PARK', 'TENDERLOIN', 'INGLESIDE', 'TARAVAL', 'CENTRAL']
    categories = ['LARCENY/THEFT', 'OTHER OFFENSES', 'ASSAULT', 'DRUG/NARCOTIC',
                  'BURGLARY', 'VANDALISM', 'STOLEN PROPERTY', 'FRAUD', 'ROBBERY',
                  'VEHICLE THEFT']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    df = pd.DataFrame({
        'dates': dates,
        'category': np.random.choice(categories, n_samples),
        'pddistrict': np.random.choice(districts, n_samples),
        'dayofweek': np.random.choice(days, n_samples),
        'resolution': np.random.choice(['ARREST', 'NONE', 'JUVENILE', 'CLEARED'], n_samples),
        'address': [f'{np.random.randint(100, 999)} {np.random.choice(["Mission", "Market", "Howard", "Folsom"])} St' for _ in range(n_samples)],
        'x': np.random.uniform(-122.52, -122.35, n_samples),
        'y': np.random.uniform(37.70, 37.82, n_samples)
    })
    
    return df

def load_and_clean():
    """Load and clean the SF crime dataset"""
    global df_raw
    
    # Try multiple possible paths
    possible_paths = [
        "data/train.csv",
        "train.csv",
        "../data/train.csv",
        "/kaggle/input/sf-crime/train.csv",
        "sf-crime/train.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Data loaded from: {path}")
            break
    
    if df is None:
        print("Training data not found. Using sample data for demonstration.")
        df = create_sample_data()
    
    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop nulls in critical columns
    df = df.dropna(subset=["dates", "pddistrict"])
    
    # Standardize string columns
    if "category" in df.columns:
        df["category"] = df["category"].str.strip().str.upper()
    df["pddistrict"] = df["pddistrict"].str.strip().str.upper()
    df["dayofweek"] = df["dayofweek"].str.strip().str.title()
    
    if "resolution" in df.columns:
        df["resolution"] = df["resolution"].str.strip().str.upper()
    
    # GPS sanity filter (if coordinates exist)
    if "x" in df.columns and "y" in df.columns:
        df = df[(df["x"].between(-123.0, -122.3)) & (df["y"].between(37.6, 37.9))]
    
    # Convert dates and create features
    df["dates"] = pd.to_datetime(df["dates"])
    df["year"] = df["dates"].dt.year
    df["month"] = df["dates"].dt.month
    df["hour"] = df["dates"].dt.hour
    df["day"] = df["dates"].dt.day
    df["week_start"] = df["dates"].dt.to_period("W").apply(lambda r: r.start_time)
    df["dayofweek_num"] = df["dates"].dt.dayofweek
    
    df_raw = df
    return df

def compute_weekly(df):
    """Compute weekly crime statistics"""
    weekly = (
        df.groupby("week_start")
        .size()
        .reset_index(name="crime_count")
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    
    weekly["prev_week"] = weekly["crime_count"].shift(1)
    weekly["delta"] = weekly["crime_count"] - weekly["prev_week"]
    weekly["pct_change"] = (weekly["delta"] / weekly["prev_week"] * 100).round(2)
    weekly["rolling_mean"] = weekly["crime_count"].rolling(4, min_periods=1).mean()
    weekly["rolling_std"] = weekly["crime_count"].rolling(4, min_periods=1).std()
    weekly["z_score"] = (weekly["crime_count"] - weekly["rolling_mean"]) / weekly["rolling_std"]
    weekly["z_score"] = weekly["z_score"].fillna(0)
    
    return weekly

def compute_districts(df):
    """Compute district statistics"""
    districts = (
        df.groupby("pddistrict")
        .agg(
            total=("category", "count"),
            types=("category", "nunique"),
            lat=("y", "mean"),
            lon=("x", "mean")
        )
        .reset_index()
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )
    districts["rank"] = range(1, len(districts) + 1)
    districts["pct"] = (districts["total"] / districts["total"].sum() * 100).round(2)
    return districts

def run_kmeans(df, k=8, n=40000):
    """Run KMeans clustering on crime coordinates"""
    sample = df[["x", "y"]].dropna().sample(min(n, len(df)), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample.values)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample = sample.copy()
    sample["cluster"] = kmeans.fit_predict(X_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    return sample, centroids

def apply_filters(df, years=None, categories=None, districts=None):
    """Apply filters to dataframe"""
    filtered = df.copy()
    
    if years:
        filtered = filtered[filtered["year"].isin(years)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if districts:
        filtered = filtered[filtered["pddistrict"].isin(districts)]
    
    return filtered


# ROUTES


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data/overview')
def api_data_overview():
    """API endpoint for data overview"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get unique values for filters
    filters = {
        'years': sorted(df_raw['year'].unique()),
        'categories': sorted(df_raw['category'].unique()),
        'districts': sorted(df_raw['pddistrict'].unique())
    }
    
    # Basic stats
    stats = {
        'total_records': len(df_raw),
        'date_min': df_raw['dates'].min().strftime('%Y-%m-%d'),
        'date_max': df_raw['dates'].max().strftime('%Y-%m-%d'),
        'num_districts': df_raw['pddistrict'].nunique(),
        'num_categories': df_raw['category'].nunique()
    }
    
    return jsonify({
        'filters': filters,
        'stats': stats
    })

@app.route('/api/analytics/weekly')
def api_weekly_analysis():
    """API endpoint for weekly analysis"""
    global df_raw, weekly_data
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts = request.args.getlist('districts')
    spike_thresh = float(request.args.get('spike_thresh', 2.0))
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None, 
                            categories if categories else None,
                            districts if districts else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Compute weekly stats
    weekly = compute_weekly(filtered)
    weekly["is_spike"] = weekly["z_score"].abs() > spike_thresh
    weekly["spike_type"] = np.where(
        weekly["z_score"] > spike_thresh, "HIGH",
        np.where(weekly["z_score"] < -spike_thresh, "LOW", "NORMAL")
    )
    
    # Prepare response
    weekly_data = weekly.copy()
    
    response = {
        'weekly_data': {
            'dates': weekly['week_start'].dt.strftime('%Y-%m-%d').tolist(),
            'crime_counts': weekly['crime_count'].tolist(),
            'rolling_means': weekly['rolling_mean'].tolist(),
            'pct_changes': weekly['pct_change'].fillna(0).tolist(),
            'z_scores': weekly['z_score'].tolist(),
            'spike_types': weekly['spike_type'].tolist()
        },
        'stats': {
            'avg_weekly': weekly['crime_count'].mean(),
            'max_weekly': weekly['crime_count'].max(),
            'min_weekly': weekly['crime_count'].min(),
            'total_weeks': len(weekly),
            'high_spikes': (weekly['z_score'] > spike_thresh).sum(),
            'low_spikes': (weekly['z_score'] < -spike_thresh).sum()
        }
    }
    
    return jsonify(response)

@app.route('/api/analytics/districts')
def api_district_analysis():
    """API endpoint for district analysis"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Compute district stats
    districts = compute_districts(filtered)
    
    response = {
        'districts': districts.to_dict('records')
    }
    
    return jsonify(response)

@app.route('/api/analytics/crime-patterns')
def api_crime_patterns():
    """API endpoint for crime pattern analysis"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Top categories
    top_categories = filtered['category'].value_counts().head(15).to_dict()
    
    # Day of week distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = filtered['dayofweek'].value_counts().reindex(day_order).fillna(0).to_dict()
    
    # Hourly distribution
    hourly = filtered.groupby('hour').size().to_dict()
    
    # Yearly trends
    yearly = filtered.groupby(['year', 'category']).size().reset_index(name='count')
    top5_cats = filtered['category'].value_counts().head(5).index.tolist()
    yearly_top = yearly[yearly['category'].isin(top5_cats)]
    yearly_trends = yearly_top.groupby('year')['count'].sum().to_dict()
    
    # Day-Hour heatmap data
    pivot = filtered.pivot_table(
        index='dayofweek',
        columns='hour',
        values='category',
        aggfunc='count',
        observed=True
    )
    pivot = pivot.reindex(day_order).fillna(0)
    
    response = {
        'top_categories': top_categories,
        'day_distribution': day_counts,
        'hourly_distribution': hourly,
        'yearly_trends': yearly_trends,
        'heatmap': {
            'days': day_order,
            'hours': list(range(24)),
            'data': pivot.values.tolist()
        }
    }
    
    return jsonify(response)

@app.route('/api/analytics/clusters')
def api_cluster_analysis():
    """API endpoint for cluster analysis"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get parameters
    k = int(request.args.get('k', 8))
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Run clustering
    sample, centroids = run_kmeans(filtered, k=k)
    
    # Get cluster statistics
    cluster_counts = sample['cluster'].value_counts().sort_index().to_dict()
    
    # Get dominant category per cluster
    cluster_categories = sample.merge(
        filtered[['x', 'y', 'category']], on=['x', 'y'], how='left'
    )
    
    dominant_cats = {}
    for i in range(k):
        cluster_cat = cluster_categories[cluster_categories['cluster'] == i]['category']
        if len(cluster_cat) > 0:
            dominant_cats[i] = cluster_cat.value_counts().index[0]
        else:
            dominant_cats[i] = 'Unknown'
    
    response = {
        'k': k,
        'cluster_counts': cluster_counts,
        'centroids': centroids.tolist(),
        'dominant_categories': dominant_cats,
        'sample_points': sample[['x', 'y', 'cluster']].head(5000).to_dict('records')
    }
    
    return jsonify(response)

@app.route('/api/map/crime-map')
def api_crime_map():
    """Generate crime map HTML"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    show_heatmap = request.args.get('show_heatmap', 'false').lower() == 'true'
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Compute district stats
    districts = compute_districts(filtered)
    
    # Create map
    m = folium.Map(location=[37.77, -122.42], zoom_start=12, tiles='CartoDB dark_matter')
    
    # Add district circles
    max_total = districts['total'].max()
    min_total = districts['total'].min()
    
    for _, row in districts.iterrows():
        radius = 8 + (row['total'] - min_total) / (max_total - min_total) * 27
        color = f"#{int(255 * (1 - row['rank']/len(districts))):02x}{int(255 * (row['rank']/len(districts))):02x}40"
        
        popup_text = f"""
        <div style="font-family: monospace; min-width: 180px;">
            <b>District {row['pddistrict']}</b><br>
            <b>Rank:</b> #{row['rank']}<br>
            <b>Total Crimes:</b> {row['total']:,}<br>
            <b>Share:</b> {row['pct']}%<br>
            <b>Crime Types:</b> {row['types']}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{row['pddistrict']}: {row['total']:,} crimes",
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=2
        ).add_to(m)
    
    # Add heatmap if requested
    if show_heatmap:
        heat_data = filtered[['y', 'x']].dropna().values.tolist()
        HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3).add_to(m)
    
    # Save map to string
    map_html = m.get_root().render()
    
    return jsonify({'map_html': map_html})

@app.route('/api/export/weekly')
def export_weekly():
    """Export weekly data as CSV"""
    global weekly_data
    
    if weekly_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    csv_data = weekly_data.to_csv(index=False)
    
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='weekly_crime_summary.csv'
    )

@app.route('/api/export/districts')
def export_districts():
    """Export district data as CSV"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available'}), 400
    
    districts = compute_districts(filtered)
    csv_data = districts.to_csv(index=False)
    
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='district_rankings.csv'
    )

@app.route('/api/export/filtered')
def export_filtered():
    """Export filtered data as CSV"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available'}), 400
    
    csv_data = filtered.to_csv(index=False)
    
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='cityguard_filtered_data.csv'
    )

@app.route('/api/insights')
def api_insights():
    """Get insights and recommendations"""
    global df_raw
    
    if df_raw is None:
        load_and_clean()
    
    # Get filter parameters
    years = request.args.getlist('years', type=int)
    categories = request.args.getlist('categories')
    districts_filter = request.args.getlist('districts')
    spike_thresh = float(request.args.get('spike_thresh', 2.0))
    
    # Apply filters
    filtered = apply_filters(df_raw, years if years else None,
                            categories if categories else None,
                            districts_filter if districts_filter else None)
    
    if len(filtered) == 0:
        return jsonify({'error': 'No data available with current filters'}), 400
    
    # Compute metrics
    total_crimes = len(filtered)
    date_range_days = (filtered['dates'].max() - filtered['dates'].min()).days
    avg_daily = total_crimes / date_range_days if date_range_days > 0 else 0
    
    districts = compute_districts(filtered)
    top_district = districts.iloc[0]['pddistrict'] if len(districts) > 0 else 'N/A'
    top_district_pct = districts.iloc[0]['pct'] if len(districts) > 0 else 0
    
    top_category = filtered['category'].value_counts().index[0] if len(filtered) > 0 else 'N/A'
    top_category_pct = filtered['category'].value_counts().iloc[0] / len(filtered) * 100
    
    peak_hour = filtered['hour'].value_counts().index[0] if len(filtered) > 0 else 0
    peak_day = filtered['dayofweek'].value_counts().index[0] if len(filtered) > 0 else 'N/A'
    
    weekly = compute_weekly(filtered)
    weekly['is_spike'] = weekly['z_score'].abs() > spike_thresh
    weekly['spike_type'] = np.where(
        weekly['z_score'] > spike_thresh, 'HIGH',
        np.where(weekly['z_score'] < -spike_thresh, 'LOW', 'NORMAL')
    )
    
    high_spikes = weekly[weekly['spike_type'] == 'HIGH']
    worst_spike = high_spikes.loc[high_spikes['pct_change'].idxmax()] if not high_spikes.empty else None
    
    recent_avg = weekly['crime_count'].tail(4).mean()
    overall_avg = weekly['crime_count'].mean()
    trend_pct = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
    
    response = {
        'stats': {
            'total_crimes': total_crimes,
            'date_range_days': date_range_days,
            'avg_daily': round(avg_daily, 1),
            'unique_crime_types': filtered['category'].nunique(),
            'num_districts': filtered['pddistrict'].nunique()
        },
        'findings': {
            'top_district': top_district,
            'top_district_pct': top_district_pct,
            'top_category': top_category,
            'top_category_pct': round(top_category_pct, 1),
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'anomaly_weeks': weekly['is_spike'].sum(),
            'high_spikes': len(high_spikes)
        },
        'trend': {
            'trend_percentage': round(trend_pct, 1),
            'recent_avg': round(recent_avg, 1),
            'overall_avg': round(overall_avg, 1)
        },
        'worst_spike': {
            'week': worst_spike['week_start'].strftime('%Y-%m-%d') if worst_spike is not None else None,
            'pct_change': worst_spike['pct_change'] if worst_spike is not None else None,
            'z_score': worst_spike['z_score'] if worst_spike is not None else None
        } if worst_spike is not None else None
    }
    
    return jsonify(response)


# CREATE TEMPLATES


def create_templates():
    """Create HTML templates"""
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Index template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CityGuard - Crime Intelligence Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #0d1117;
            color: #cdd9e5;
            font-family: 'Segoe UI', sans-serif;
        }
        .sidebar {
            background-color: #161b22;
            border-right: 1px solid #30363d;
            min-height: 100vh;
            padding: 20px;
        }
        .main-content {
            padding: 20px;
        }
        .card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #1c2128;
            border-bottom: 1px solid #30363d;
            color: #58a6ff;
            font-weight: bold;
        }
        .metric-card {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #1c2128 0%, #161b22 100%);
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #58a6ff;
        }
        .metric-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
        }
        .nav-tabs {
            border-bottom-color: #30363d;
        }
        .nav-tabs .nav-link {
            color: #8b949e;
            background-color: transparent;
            border: none;
        }
        .nav-tabs .nav-link.active {
            color: #58a6ff;
            background-color: transparent;
            border-bottom: 2px solid #58a6ff;
        }
        .btn-primary {
            background-color: #58a6ff;
            border-color: #58a6ff;
        }
        .btn-primary:hover {
            background-color: #79c0ff;
            border-color: #79c0ff;
        }
        select, input {
            background-color: #0d1117 !important;
            color: #cdd9e5 !important;
            border-color: #30363d !important;
        }
        .alert-red {
            background: #2d1215;
            border-left: 4px solid #ff6b6b;
            padding: 12px 16px;
            margin: 6px 0;
            color: #ff6b6b;
        }
        .alert-blue {
            background: #1a2433;
            border-left: 4px solid #58a6ff;
            padding: 12px 16px;
            margin: 6px 0;
            color: #58a6ff;
        }
        .alert-green {
            background: #12261e;
            border-left: 4px solid #3fb950;
            padding: 12px 16px;
            margin: 6px 0;
            color: #3fb950;
        }
        .loading {
            text-align: center;
            padding: 50px;
        }
        .spinner {
            border: 3px solid #30363d;
            border-top: 3px solid #58a6ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        iframe {
            width: 100%;
            height: 500px;
            border: none;
            border-radius: 8px;
        }
        .badge-red {
            background: #3d1e1e;
            color: #ff6b6b;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
        }
        .badge-blue {
            background: #1e2e3d;
            color: #58a6ff;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
        }
        .badge-green {
            background: #1e3d2e;
            color: #3fb950;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar">
                <h3> CityGuard</h3>
                <p class="small text-muted">Crime Intelligence Platform</p>
                <hr>
                
                <div class="mb-3">
                    <label class="form-label"> Year Filter</label>
                    <select id="yearFilter" class="form-select" multiple size="3">
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label"> Crime Category</label>
                    <select id="categoryFilter" class="form-select" multiple size="3">
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label"> District</label>
                    <select id="districtFilter" class="form-select" multiple size="3">
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label"> Spike Threshold (Z-score)</label>
                    <input type="range" id="spikeThreshold" class="form-range" min="1" max="4" step="0.1" value="2.0">
                    <span id="thresholdValue" class="badge bg-info mt-2">2.0</span>
                </div>
                
                <hr>
                <div class="small text-muted">
                    <div id="statsInfo">Loading data...</div>
                </div>
                
                <button class="btn btn-primary btn-sm w-100 mt-3" onclick="applyFilters()">
                     Apply Filters
                </button>
                <button class="btn btn-secondary btn-sm w-100 mt-2" onclick="resetFilters()">
                     Reset
                </button>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 main-content">
                <!-- KPIs -->
                <div class="row" id="kpiRow">
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="totalCrimes">-</div>
                            <div class="metric-label">Total Crimes</div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="weeksTracked">-</div>
                            <div class="metric-label">Weeks Tracked</div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="anomalyWeeks">-</div>
                            <div class="metric-label">Anomaly Weeks</div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="topDistrict">-</div>
                            <div class="metric-label">Top District</div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="topCategory">-</div>
                            <div class="metric-label">Top Category</div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="peakHour">-</div>
                            <div class="metric-label">Peak Hour</div>
                        </div>
                    </div>
                </div>
                
                <!-- Tabs -->
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#weekly">📈 Weekly Trends</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#geographic">🗺️ Geographic Analysis</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#patterns">📊 Crime Patterns</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#clusters">🤖 Cluster Analysis</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#insights">💡 Insights</button>
                    </li>
                </ul>
                
                <div class="tab-content mt-3">
                    <!-- Weekly Trends Tab -->
                    <div class="tab-pane fade show active" id="weekly">
                        <div class="card">
                            <div class="card-header">Weekly Crime Trends</div>
                            <div class="card-body">
                                <div id="weeklyChart"></div>
                                <div id="weeklyChangeChart"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Geographic Analysis Tab -->
                    <div class="tab-pane fade" id="geographic">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="card">
                                    <div class="card-header">Interactive Crime Map</div>
                                    <div class="card-body">
                                        <div class="form-check mb-2">
                                            <input class="form-check-input" type="checkbox" id="heatmapToggle">
                                            <label class="form-check-label">Show Heatmap Overlay</label>
                                        </div>
                                        <iframe id="crimeMap" srcdoc="<div class='loading'><div class='spinner'></div><p>Loading map...</p></div>"></iframe>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-header">District Rankings</div>
                                    <div class="card-body">
                                        <div id="districtRankings"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Crime Patterns Tab -->
                    <div class="tab-pane fade" id="patterns">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">Top Crime Categories</div>
                                    <div class="card-body">
                                        <div id="topCategoriesChart"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">Crime by Day of Week</div>
                                    <div class="card-body">
                                        <div id="dayOfWeekChart"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">Day × Hour Heatmap</div>
                                    <div class="card-body">
                                        <div id="heatmapChart"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Cluster Analysis Tab -->
                    <div class="tab-pane fade" id="clusters">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="card">
                                    <div class="card-header">Cluster Settings</div>
                                    <div class="card-body">
                                        <label>Number of Clusters (K)</label>
                                        <input type="range" id="kValue" class="form-range" min="3" max="12" step="1" value="8">
                                        <span id="kValueDisplay" class="badge bg-info mt-2">8</span>
                                        <button class="btn btn-primary btn-sm w-100 mt-3" onclick="updateClusters()">
                                            🔄 Update Clusters
                                        </button>
                                    </div>
                                </div>
                                <div class="card mt-3">
                                    <div class="card-header">Cluster Statistics</div>
                                    <div class="card-body">
                                        <div id="clusterStats"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-9">
                                <div class="card">
                                    <div class="card-header">Crime Hotspot Clusters</div>
                                    <div class="card-body">
                                        <div id="clusterChart"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Insights Tab -->
                    <div class="tab-pane fade" id="insights">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">Key Statistics</div>
                                    <div class="card-body" id="insightsStats">
                                        <div class="loading">Loading insights...</div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">Critical Findings</div>
                                    <div class="card-body" id="criticalFindings">
                                        <div class="loading">Loading findings...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">Recommended Actions</div>
                                    <div class="card-body" id="recommendations">
                                        <div class="loading">Loading recommendations...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentFilters = {
            years: [],
            categories: [],
            districts: [],
            spike_thresh: 2.0
        };
        
        // Load initial data
        $(document).ready(function() {
            loadFilters();
            applyFilters();
            
            $('#spikeThreshold').on('input', function() {
                $('#thresholdValue').text($(this).val());
            });
            
            $('#kValue').on('input', function() {
                $('#kValueDisplay').text($(this).val());
            });
            
            $('#heatmapToggle').on('change', function() {
                updateMap();
            });
        });
        
        function loadFilters() {
            $.get('/api/data/overview', function(data) {
                // Populate year filter
                let yearSelect = $('#yearFilter');
                yearSelect.empty();
                data.filters.years.forEach(year => {
                    yearSelect.append(`<option value="${year}">${year}</option>`);
                });
                
                // Populate category filter
                let catSelect = $('#categoryFilter');
                catSelect.empty();
                data.filters.categories.forEach(cat => {
                    catSelect.append(`<option value="${cat}">${cat}</option>`);
                });
                
                // Populate district filter
                let distSelect = $('#districtFilter');
                distSelect.empty();
                data.filters.districts.forEach(dist => {
                    distSelect.append(`<option value="${dist}">${dist}</option>`);
                });
                
                $('#statsInfo').html(`
                    📊 ${data.stats.total_records.toLocaleString()} records<br>
                    📅 ${data.stats.date_min} → ${data.stats.date_max}<br>
                    🏢 ${data.stats.num_districts} districts
                `);
            });
        }
        
        function applyFilters() {
            currentFilters.years = $('#yearFilter').val() || [];
            currentFilters.categories = $('#categoryFilter').val() || [];
            currentFilters.districts = $('#districtFilter').val() || [];
            currentFilters.spike_thresh = parseFloat($('#spikeThreshold').val());
            
            updateWeeklyAnalysis();
            updateDistrictAnalysis();
            updateCrimePatterns();
            updateInsights();
            updateMap();
        }
        
        function resetFilters() {
            $('#yearFilter').val([]);
            $('#categoryFilter').val([]);
            $('#districtFilter').val([]);
            $('#spikeThreshold').val(2.0);
            $('#thresholdValue').text('2.0');
            applyFilters();
        }
        
        function updateWeeklyAnalysis() {
            $.get('/api/analytics/weekly', currentFilters, function(data) {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                
                // Update KPIs
                $('#weeksTracked').text(data.stats.total_weeks);
                $('#anomalyWeeks').text(data.stats.high_spikes + data.stats.low_spikes);
                
                // Weekly trend chart
                let trace1 = {
                    x: data.weekly_data.dates,
                    y: data.weekly_data.crime_counts,
                    mode: 'lines',
                    name: 'Weekly Crimes',
                    line: {color: '#58a6ff', width: 2},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(88, 166, 255, 0.1)'
                };
                
                let trace2 = {
                    x: data.weekly_data.dates,
                    y: data.weekly_data.rolling_means,
                    mode: 'lines',
                    name: '4-Week Rolling Avg',
                    line: {color: '#f0db4f', width: 2, dash: 'dash'}
                };
                
                let highSpikes = {
                    x: [],
                    y: []
                };
                let lowSpikes = {
                    x: [],
                    y: []
                };
                
                for (let i = 0; i < data.weekly_data.dates.length; i++) {
                    if (data.weekly_data.spike_types[i] === 'HIGH') {
                        highSpikes.x.push(data.weekly_data.dates[i]);
                        highSpikes.y.push(data.weekly_data.crime_counts[i]);
                    } else if (data.weekly_data.spike_types[i] === 'LOW') {
                        lowSpikes.x.push(data.weekly_data.dates[i]);
                        lowSpikes.y.push(data.weekly_data.crime_counts[i]);
                    }
                }
                
                let trace3 = {
                    x: highSpikes.x,
                    y: highSpikes.y,
                    mode: 'markers',
                    name: 'HIGH Spikes',
                    marker: {color: '#ff6b6b', size: 12, symbol: 'triangle-up'}
                };
                
                let trace4 = {
                    x: lowSpikes.x,
                    y: lowSpikes.y,
                    mode: 'markers',
                    name: 'LOW Spikes',
                    marker: {color: '#4ecdc4', size: 12, symbol: 'triangle-down'}
                };
                
                let layout = {
                    title: 'Weekly Crime Trends with Anomaly Detection',
                    xaxis: {title: 'Date'},
                    yaxis: {title: 'Crime Count'},
                    template: 'plotly_dark',
                    height: 500
                };
                
                Plotly.newPlot('weeklyChart', [trace1, trace2, trace3, trace4], layout);
                
                // Percentage change chart
                let barColors = data.weekly_data.pct_changes.map(p => p > 0 ? '#ff6b6b' : '#4ecdc4');
                
                let barTrace = {
                    x: data.weekly_data.dates,
                    y: data.weekly_data.pct_changes,
                    type: 'bar',
                    name: '% Change',
                    marker: {color: barColors}
                };
                
                let barLayout = {
                    title: 'Week-over-Week Percentage Change',
                    xaxis: {title: 'Date'},
                    yaxis: {title: 'Percentage Change (%)'},
                    template: 'plotly_dark',
                    height: 400,
                    shapes: [
                        {type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper', line: {color: '#8b949e', dash: 'dash'}}
                    ]
                };
                
                Plotly.newPlot('weeklyChangeChart', [barTrace], barLayout);
            });
        }
        
        function updateDistrictAnalysis() {
            $.get('/api/analytics/districts', currentFilters, function(data) {
                if (data.error) return;
                
                // District ranking chart
                let districts = data.districts;
                let trace = {
                    x: districts.map(d => d.total),
                    y: districts.map(d => d.pddistrict),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                        color: districts.map(d => d.rank),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: {title: 'Rank'}
                    },
                    text: districts.map(d => d.pct),
                    textposition: 'outside'
                };
                
                let layout = {
                    title: 'District Crime Rankings',
                    xaxis: {title: 'Total Crimes'},
                    yaxis: {title: 'Police District'},
                    template: 'plotly_dark',
                    height: 500,
                    yaxis: {categoryorder: 'total ascending'}
                };
                
                Plotly.newPlot('districtRankings', [trace], layout);
                
                // Update top district KPI
                if (districts.length > 0) {
                    $('#topDistrict').text(districts[0].pddistrict);
                }
            });
        }
        
        function updateCrimePatterns() {
            $.get('/api/analytics/crime-patterns', currentFilters, function(data) {
                if (data.error) return;
                
                // Top categories chart
                let categories = Object.entries(data.top_categories).slice(0, 10);
                let catTrace = {
                    x: categories.map(c => c[1]),
                    y: categories.map(c => c[0]),
                    type: 'bar',
                    orientation: 'h',
                    marker: {color: 'lightsalmon'},
                    text: categories.map(c => c[1]),
                    textposition: 'outside'
                };
                
                let catLayout = {
                    title: 'Top 10 Crime Categories',
                    xaxis: {title: 'Number of Incidents'},
                    template: 'plotly_dark',
                    height: 500
                };
                
                Plotly.newPlot('topCategoriesChart', [catTrace], catLayout);
                
                // Day of week chart
                let days = Object.entries(data.day_distribution);
                let dayTrace = {
                    x: days.map(d => d[0]),
                    y: days.map(d => d[1]),
                    type: 'bar',
                    marker: {color: 'lightgreen'},
                    text: days.map(d => d[1]),
                    textposition: 'outside'
                };
                
                let dayLayout = {
                    title: 'Crime by Day of Week',
                    xaxis: {title: 'Day'},
                    yaxis: {title: 'Number of Incidents'},
                    template: 'plotly_dark',
                    height: 500
                };
                
                Plotly.newPlot('dayOfWeekChart', [dayTrace], dayLayout);
                
                // Heatmap
                let heatmapTrace = {
                    z: data.heatmap.data,
                    x: data.heatmap.hours,
                    y: data.heatmap.days,
                    type: 'heatmap',
                    colorscale: 'YlOrRd',
                    colorbar: {title: 'Crime Count'}
                };
                
                let heatmapLayout = {
                    title: 'Day × Hour Crime Heatmap',
                    xaxis: {title: 'Hour of Day'},
                    yaxis: {title: 'Day of Week'},
                    template: 'plotly_dark',
                    height: 450
                };
                
                Plotly.newPlot('heatmapChart', [heatmapTrace], heatmapLayout);
                
                // Update top category KPI
                if (categories.length > 0) {
                    $('#topCategory').text(categories[0][0]);
                }
            });
        }
        
        function updateClusters() {
            let k = $('#kValue').val();
            let params = {...currentFilters, k: k};
            
            $.get('/api/analytics/clusters', params, function(data) {
                if (data.error) return;
                
                // Cluster visualization
                let traces = [];
                let colors = ['#58a6ff', '#ff6b6b', '#3fb950', '#f0db4f', '#d2a8ff', '#ffa657', '#a5d6ff', '#7ee787', '#e3b341', '#bc8cff', '#79c0ff', '#ff8c94'];
                
                for (let i = 0; i < data.k; i++) {
                    let clusterPoints = data.sample_points.filter(p => p.cluster === i);
                    traces.push({
                        x: clusterPoints.map(p => p.x),
                        y: clusterPoints.map(p => p.y),
                        mode: 'markers',
                        name: `Cluster ${i}`,
                        marker: {size: 3, opacity: 0.5, color: colors[i % colors.length]}
                    });
                }
                
                traces.push({
                    x: data.centroids.map(c => c[0]),
                    y: data.centroids.map(c => c[1]),
                    mode: 'markers',
                    name: 'Centroids',
                    marker: {size: 15, symbol: 'star', color: 'red', line: {width: 2, color: 'white'}}
                });
                
                let layout = {
                    title: `Crime Hotspot Clusters (K=${data.k})`,
                    xaxis: {title: 'Longitude'},
                    yaxis: {title: 'Latitude'},
                    template: 'plotly_dark',
                    height: 600
                };
                
                Plotly.newPlot('clusterChart', traces, layout);
                
                // Cluster statistics
                let statsHtml = '<table class="table table-sm">';
                statsHtml += '<thead><tr><th>Cluster</th><th>Count</th><th>Dominant Crime</th></tr></thead><tbody>';
                for (let i = 0; i < data.k; i++) {
                    statsHtml += `<tr>
                        <td><span class="badge-blue">Cluster ${i}</span></td>
                        <td>${data.cluster_counts[i].toLocaleString()}</td>
                        <td>${data.dominant_categories[i]}</td>
                    </tr>`;
                }
                statsHtml += '</tbody></table>';
                $('#clusterStats').html(statsHtml);
            });
        }
        
        function updateInsights() {
            $.get('/api/insights', currentFilters, function(data) {
                if (data.error) return;
                
                // Update total crimes KPI
                $('#totalCrimes').text(data.stats.total_crimes.toLocaleString());
                
                // Update peak hour KPI
                $('#peakHour').text(`${data.findings.peak_hour}:00`);
                
                // Key statistics
                let statsHtml = `
                    <p><strong>Total Incidents:</strong> ${data.stats.total_crimes.toLocaleString()}</p>
                    <p><strong>Analysis Period:</strong> ${data.stats.date_range_days} days</p>
                    <p><strong>Average Daily Crimes:</strong> ${data.stats.avg_daily}</p>
                    <p><strong>Unique Crime Types:</strong> ${data.stats.unique_crime_types}</p>
                    <p><strong>Police Districts:</strong> ${data.stats.num_districts}</p>
                `;
                $('#insightsStats').html(statsHtml);
                
                // Critical findings
                let findingsHtml = `
                    <div class="alert-red">
                        <strong>Highest Crime District:</strong> ${data.findings.top_district} (${data.findings.top_district_pct}% of all crimes)
                    </div>
                    <div class="alert-blue">
                        <strong>Most Common Crime:</strong> ${data.findings.top_category} (${data.findings.top_category_pct}% of incidents)
                    </div>
                    <div class="alert-blue">
                        <strong>Peak Crime Window:</strong> ${data.findings.peak_hour}:00 on ${data.findings.peak_day}s
                    </div>
                    <div class="alert-red">
                        <strong>Anomaly Weeks Detected:</strong> ${data.findings.anomaly_weeks} (${data.findings.high_spikes} HIGH spikes)
                    </div>
                `;
                
                if (data.worst_spike && data.worst_spike.week) {
                    findingsHtml += `
                        <div class="alert-red">
                            <strong>Worst Spike:</strong> Week of ${data.worst_spike.week}<br>
                            Increase: +${data.worst_spike.pct_change}% (Z-score: ${data.worst_spike.z_score})
                        </div>
                    `;
                }
                
                $('#criticalFindings').html(findingsHtml);
                
                // Recommendations
                let trendIcon = data.trend.trend_percentage > 0 ? '' : '';
                let recHtml = `
                    <div class="alert-red">
                        <strong> IMMEDIATE ACTIONS</strong><br>
                        • Increase patrol presence in <strong>${data.findings.top_district}</strong><br>
                        • Focus on <strong>${data.findings.peak_hour}:00-${(data.findings.peak_hour+2)%24}:00</strong> window<br>
                        • Deploy resources to <strong>${data.findings.top_category}</strong> hotspots
                    </div>
                    <div class="alert-blue">
                        <strong> MONITORING PROTOCOL</strong><br>
                        • Track weekly Z-scores (alert at >${currentFilters.spike_thresh})<br>
                        • Review top 5 districts weekly<br>
                        • Monitor crime type trends (${trendIcon} ${Math.abs(data.trend.trend_percentage)}% recent trend)
                    </div>
                    <div class="alert-green">
                        <strong> STRATEGIC PLANNING</strong><br>
                        • Re-evaluate clusters quarterly<br>
                        • Adjust patrol zones based on hotspots<br>
                        • Share intelligence with ${data.findings.top_district} district
                    </div>
                `;
                $('#recommendations').html(recHtml);
            });
        }
        
        function updateMap() {
            let params = {...currentFilters};
            params.show_heatmap = $('#heatmapToggle').is(':checked');
            
            $.get('/api/map/crime-map', params, function(data) {
                if (data.error) return;
                $('#crimeMap').attr('srcdoc', data.map_html);
            });
        }
    </script>
</body>
</html>
"""
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)


# MAIN


if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Load data
    print("Loading data...")
    load_and_clean()
    print(f"Data loaded: {len(df_raw)} records")
    
    
    print("🏛️ CityGuard Crime Intelligence Dashboard")
   
    print("Flask application starting...")
    print("Access the dashboard at: http://localhost:5000")

    
    app.run(debug=True, host='0.0.0.0', port=5000)