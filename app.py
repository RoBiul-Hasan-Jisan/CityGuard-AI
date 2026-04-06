"""
CityGuard — Crime Intelligence Dashboard

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" CityGuard Crime Intelligence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL DARK THEME STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #cdd9e5;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Cards / metric */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; }
    [data-testid="stMetricValue"] { color: #cdd9e5 !important; font-size: 26px; font-weight: bold; }
    /* Section headers */
    h1 { color: #58a6ff !important; }
    h2 { color: #e6edf3 !important; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
    h3 { color: #cdd9e5 !important; }
    /* Dividers */
    hr { border-color: #30363d; }
    /* Tabs */
    [data-testid="stTab"] { color: #8b949e; }
    button[data-baseweb="tab"] { color: #8b949e !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }
    /* Sidebar select */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiselect label,
    [data-testid="stSidebar"] .stSlider label { color: #8b949e; }
    /* Alerts */
    .alert-red   { background:#2d1215; border-left:4px solid #ff6b6b; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#ff6b6b; }
    .alert-blue  { background:#1a2433; border-left:4px solid #58a6ff; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#58a6ff; }
    .alert-green { background:#12261e; border-left:4px solid #3fb950; padding:12px 16px; border-radius:0 6px 6px 0; margin:6px 0; color:#3fb950; }
    .badge-red   { background:#3d1e1e; color:#ff6b6b; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
    .badge-blue  { background:#1e2e3d; color:#58a6ff; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
    .badge-green { background:#1e3d2e; color:#3fb950; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:bold; }
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #161b22; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#cdd9e5",
    "grid.color": "#21262d",
    "lines.color": "#58a6ff",
    "savefig.facecolor": "#0d1117",
})

# ─────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_and_clean():
    """Load and clean the SF crime dataset"""
    # Try multiple possible paths
    possible_paths = [
        "../data/train.csv",
        "data/train.csv",
        "train.csv",
        "/kaggle/input/sf-crime/train.csv",
        "sf-crime/train.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.success(f" Data loaded from: {path}")
            break
    
    if df is None:
        # Create sample data for demonstration if file not found
        st.warning(" Training data not found. Using sample data for demonstration.")
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
    
    return df

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

@st.cache_data(show_spinner=False)
def compute_weekly(_df):
    """Compute weekly crime statistics"""
    weekly = (
        _df.groupby("week_start")
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

@st.cache_data(show_spinner=False)
def compute_districts(_df):
    """Compute district statistics"""
    districts = (
        _df.groupby("pddistrict")
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

@st.cache_data(show_spinner=False)
def run_kmeans(_df, k=8, n=40000):
    """Run KMeans clustering on crime coordinates"""
    sample = _df[["x", "y"]].dropna().sample(min(n, len(_df)), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample.values)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample = sample.copy()
    sample["cluster"] = kmeans.fit_predict(X_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    return sample, centroids

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  CityGuard")
    st.markdown("*Crime Intelligence Platform*")
    st.markdown("---")

    with st.spinner("Loading data..."):
        df_raw = load_and_clean()

    years = sorted(df_raw["year"].unique())
    sel_years = st.multiselect(" Year filter", years, default=years[:3] if len(years) > 3 else years)

    all_cats = sorted(df_raw["category"].unique())
    sel_cats = st.multiselect(" Crime category", all_cats, default=[])

    all_dists = sorted(df_raw["pddistrict"].unique())
    sel_dists = st.multiselect(" District", all_dists, default=[])

    spike_thresh = st.slider("Spike Z-score threshold", 1.0, 4.0, 2.0, 0.1)

    st.markdown("---")
    st.caption(f" Dataset: **{len(df_raw):,}** records")
    st.caption(f" {df_raw['dates'].min().date()} → {df_raw['dates'].max().date()}")
    st.caption(f" {df_raw['pddistrict'].nunique()} districts |  {df_raw['category'].nunique()} crime types")

# Apply filters
df = df_raw.copy()
if sel_years:
    df = df[df["year"].isin(sel_years)]
if sel_cats:
    df = df[df["category"].isin(sel_cats)]
if sel_dists:
    df = df[df["pddistrict"].isin(sel_dists)]

if len(df) == 0:
    st.error(" No data available with current filters. Please adjust your selection.")
    st.stop()

weekly = compute_weekly(df)
weekly["is_spike"] = weekly["z_score"].abs() > spike_thresh
weekly["spike_type"] = np.where(
    weekly["z_score"] > spike_thresh, "HIGH",
    np.where(weekly["z_score"] < -spike_thresh, "LOW", "NORMAL")
)

districts = compute_districts(df)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("#  CityGuard — Crime Intelligence Dashboard")
st.markdown("*San Francisco Police Department Crime Data — Powered by AI*")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(" Total Crimes", f"{len(df):,}", delta=f"Filtered")
with col2:
    st.metric(" Weeks Tracked", f"{len(weekly):,}")
with col3:
    st.metric(" Anomalous Weeks", f"{weekly['is_spike'].sum()}", 
              delta=f"{(weekly['is_spike'].sum()/len(weekly)*100):.1f}%")
with col4:
    top_dist = districts.iloc[0]["pddistrict"] if len(districts) > 0 else "N/A"
    st.metric(" Top District", top_dist)
with col5:
    top_cat = df["category"].value_counts().index[0] if len(df) > 0 else "N/A"
    st.metric(" Top Category", top_cat)
with col6:
    peak_hour = df["hour"].value_counts().index[0] if len(df) > 0 else 0
    st.metric(" Peak Hour", f"{peak_hour:02d}:00")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Weekly Trends",
    " Geographic Analysis",
    " Crime Patterns",
    " Cluster Analysis",
    " Insights & Actions",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — WEEKLY TRENDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("Weekly Crime Volume Analysis")
    
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Weekly Crimes", f"{weekly['crime_count'].mean():,.0f}")
    with c2:
        st.metric("Max Weekly Crimes", f"{weekly['crime_count'].max():,}")
    with c3:
        st.metric("Min Weekly Crimes", f"{weekly['crime_count'].min():,}")
    with c4:
        high_spikes = (weekly["z_score"] > spike_thresh).sum()
        st.metric("HIGH Spikes", high_spikes, delta=f"z > {spike_thresh}")
    
    # Weekly trend chart using Plotly for interactivity
    fig = go.Figure()
    
    # Add weekly crime line
    fig.add_trace(go.Scatter(
        x=weekly["week_start"],
        y=weekly["crime_count"],
        mode="lines",
        name="Weekly Crimes",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88, 166, 255, 0.1)"
    ))
    
    # Add rolling average
    fig.add_trace(go.Scatter(
        x=weekly["week_start"],
        y=weekly["rolling_mean"],
        mode="lines",
        name="4-Week Rolling Avg",
        line=dict(color="#f0db4f", width=2, dash="dash")
    ))
    
    # Add spike markers
    high_spikes_data = weekly[weekly["spike_type"] == "HIGH"]
    low_spikes_data = weekly[weekly["spike_type"] == "LOW"]
    
    fig.add_trace(go.Scatter(
        x=high_spikes_data["week_start"],
        y=high_spikes_data["crime_count"],
        mode="markers",
        name="HIGH Spikes",
        marker=dict(color="#ff6b6b", size=12, symbol="triangle-up")
    ))
    
    fig.add_trace(go.Scatter(
        x=low_spikes_data["week_start"],
        y=low_spikes_data["crime_count"],
        mode="markers",
        name="LOW Spikes",
        marker=dict(color="#4ecdc4", size=12, symbol="triangle-down")
    ))
    
    fig.update_layout(
        title="Weekly Crime Trends with Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Crime Count",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Percentage change bar chart
    fig2 = go.Figure()
    
    colors = ["#ff6b6b" if x > 0 else "#4ecdc4" for x in weekly["pct_change"].fillna(0)]
    
    fig2.add_trace(go.Bar(
        x=weekly["week_start"],
        y=weekly["pct_change"].fillna(0),
        name="% Change",
        marker_color=colors,
        hovertemplate="Week: %{x}<br>Change: %{y:.1f}%<extra></extra>"
    ))
    
    fig2.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    fig2.add_hline(y=spike_thresh*10, line_dash="dot", line_color="#ff6b6b", 
                   annotation_text=f"+{spike_thresh*10}% Warning")
    fig2.add_hline(y=-spike_thresh*10, line_dash="dot", line_color="#4ecdc4",
                   annotation_text=f"-{spike_thresh*10}% Warning")
    
    fig2.update_layout(
        title="Week-over-Week Percentage Change",
        xaxis_title="Date",
        yaxis_title="Percentage Change (%)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Spike table
    with st.expander(" Detailed Spike Weeks Data", expanded=False):
        spikes_df = weekly[weekly["is_spike"]].sort_values("z_score", key=abs, ascending=False)
        if not spikes_df.empty:
            display_cols = ["week_start", "crime_count", "prev_week", "pct_change", "z_score", "spike_type"]
            st.dataframe(spikes_df[display_cols].style.format({
                "pct_change": "{:.1f}%",
                "z_score": "{:.2f}"
            }), use_container_width=True)
        else:
            st.info("No anomalous weeks found with current filter/threshold.")
    
    # Download button
    st.download_button(
        " Download Weekly Summary (CSV)",
        weekly.to_csv(index=False).encode(),
        "weekly_crime_summary.csv",
        "text/csv"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — GEOGRAPHIC ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("District Crime Analysis")
    
    # District ranking chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=districts["total"],
        y=districts["pddistrict"],
        orientation="h",
        marker=dict(
            color=districts["rank"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Rank")
        ),
        text=districts["pct"],
        textposition="outside",
        hovertemplate="District: %{y}<br>Crimes: %{x:,}<br>Share: %{text:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="District Crime Rankings",
        xaxis_title="Total Crimes",
        yaxis_title="Police District",
        template="plotly_dark",
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # District metrics table
    col_map, col_stats = st.columns([0.6, 0.4])
    
    with col_stats:
        st.markdown("### District Rankings")
        st.dataframe(
            districts[["rank", "pddistrict", "total", "pct", "types"]].head(10),
            use_container_width=True,
            hide_index=True
        )
    
    with col_map:
        st.markdown("### Interactive Crime Map")
        
        # Create base map
        m = folium.Map(location=[37.77, -122.42], zoom_start=12, tiles="CartoDB dark_matter")
        
        # Add district circles
        max_total = districts["total"].max()
        min_total = districts["total"].min()
        
        for _, row in districts.iterrows():
            radius = 8 + (row["total"] - min_total) / (max_total - min_total) * 27
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
                location=[row["lat"], row["lon"]],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"{row['pddistrict']}: {row['total']:,} crimes",
                color=color,
                fill=True,
                fill_opacity=0.6,
                weight=2
            ).add_to(m)
        
        # Heatmap toggle
        show_heat = st.checkbox("Show crime heatmap", value=False, key="heatmap_toggle")
        if show_heat:
            heat_data = df[["y", "x"]].dropna().values.tolist()
            HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3).add_to(m)
        
        st_folium(m, width=None, height=500, returned_objects=[])
    
    # Crime web visualization
    st.markdown("### Crime Category Web")
    st.caption("Connecting districts that share the same top crime categories")
    
    # Get top categories per district
    dist_top = df.groupby(["pddistrict", "category"]).size().reset_index(name="count")
    dist_top = dist_top.sort_values("count", ascending=False).groupby("pddistrict").first().reset_index()
    
    top_web_cats = dist_top["category"].value_counts().head(5).index.tolist()
    
    # Create network visualization
    for cat in top_web_cats:
        districts_with_cat = dist_top[dist_top["category"] == cat]["pddistrict"].tolist()
        if len(districts_with_cat) > 1:
            st.write(f"**{cat}** - Connected districts: {', '.join(districts_with_cat)}")
    
    # Download district data
    st.download_button(
        " Download District Rankings (CSV)",
        districts.to_csv(index=False).encode(),
        "district_rankings.csv",
        "text/csv"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — CRIME PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("Crime Pattern Analysis")
    
    # Top categories chart
    col1, col2 = st.columns(2)
    
    with col1:
        top_n = st.slider("Number of top categories", 5, 20, 10, key="top_n_slider")
        top_cats = df["category"].value_counts().head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_cats.values,
                y=top_cats.index,
                orientation="h",
                marker_color="lightsalmon",
                text=top_cats.values,
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Crime Categories",
            xaxis_title="Number of Incidents",
            yaxis_title="Category",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = df["dayofweek"].value_counts().reindex(day_order).fillna(0)
        
        fig = go.Figure(data=[
            go.Bar(
                x=day_counts.index,
                y=day_counts.values,
                marker_color="lightgreen",
                text=day_counts.values,
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            title="Crime Distribution by Day of Week",
            xaxis_title="Day",
            yaxis_title="Number of Incidents",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Hourly distribution
    st.subheader("Hourly Crime Patterns")
    
    hourly = df.groupby("hour").size().reset_index(name="count")
    
    fig = go.Figure(data=[
        go.Scatter(
            x=hourly["hour"],
            y=hourly["count"],
            mode="lines+markers",
            marker=dict(size=8, color="#58a6ff"),
            line=dict(width=2, color="#58a6ff"),
            fill="tozeroy"
        )
    ])
    
    fig.update_layout(
        title="Crime Distribution by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Number of Incidents",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day-Hour heatmap
    st.subheader("Day × Hour Crime Heatmap")
    
    pivot = df.pivot_table(
        index="dayofweek",
        columns="hour",
        values="category",
        aggfunc="count",
        observed=True
    )
    pivot = pivot.reindex(day_order)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
        title="Crime Frequency Heatmap",
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly trends
    st.subheader("Yearly Crime Trends")
    
    yearly = df.groupby(["year", "category"]).size().reset_index(name="count")
    top5_cats = df["category"].value_counts().head(5).index.tolist()
    yearly_top = yearly[yearly["category"].isin(top5_cats)]
    
    fig = go.Figure()
    
    colors = ["#58a6ff", "#ff6b6b", "#3fb950", "#f0db4f", "#d2a8ff"]
    for i, cat in enumerate(top5_cats):
        cat_data = yearly_top[yearly_top["category"] == cat]
        fig.add_trace(go.Scatter(
            x=cat_data["year"],
            y=cat_data["count"],
            mode="lines+markers",
            name=cat[:20],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Top 5 Crime Categories - Year-over-Year Trends",
        xaxis_title="Year",
        yaxis_title="Number of Incidents",
        template="plotly_dark",
        height=450,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — CLUSTER ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.subheader("Geographical Crime Hotspot Clustering")
    
    k_val = st.slider("Number of Clusters (K)", 3, 12, 8, key="kmeans_slider")
    
    with st.spinner(f"Running K-Means clustering with K={k_val}..."):
        sample_km, centroids = run_kmeans(df, k=k_val)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cluster Visualization")
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i in range(k_val):
            cluster_data = sample_km[sample_km["cluster"] == i]
            fig.add_trace(go.Scatter(
                x=cluster_data["x"],
                y=cluster_data["y"],
                mode="markers",
                name=f"Cluster {i}",
                marker=dict(size=3, opacity=0.5, color=colors[i % len(colors)]),
                hovertemplate="Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>"
            ))
        
        # Add centroids
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            name="Centroids",
            marker=dict(size=15, symbol="star", color="red", line=dict(width=2, color="white"))
        ))
        
        fig.update_layout(
            title=f"Crime Hotspot Clusters (K={k_val})",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Cluster Statistics")
        
        cluster_counts = sample_km["cluster"].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                marker_color=colors[:len(cluster_counts)],
                text=cluster_counts.values,
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            title="Incidents per Cluster",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Incidents (Sample)",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Centroid Coordinates")
        centroid_df = pd.DataFrame(centroids, columns=["Longitude", "Latitude"])
        centroid_df.index.name = "Cluster"
        st.dataframe(centroid_df.round(5), use_container_width=True)
        
        # Cluster interpretation
        st.markdown("### Cluster Interpretation")
        
        # Find dominant category per cluster
        cluster_categories = sample_km.merge(
            df[["x", "y", "category"]], on=["x", "y"], how="left"
        )
        
        dominant_cats = {}
        for i in range(k_val):
            cluster_cat = cluster_categories[cluster_categories["cluster"] == i]["category"]
            if len(cluster_cat) > 0:
                dominant_cats[i] = cluster_cat.value_counts().index[0]
        
        for cluster_id, top_cat in list(dominant_cats.items())[:5]:
            st.write(f"**Cluster {cluster_id}**: Dominated by *{top_cat}*")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — INSIGHTS & ACTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.subheader("Intelligence Summary & Recommendations")
    
    # Key metrics for insights
    total_crimes = len(df)
    date_range_days = (df["dates"].max() - df["dates"].min()).days
    avg_daily = total_crimes / date_range_days if date_range_days > 0 else 0
    
    top_district = districts.iloc[0]["pddistrict"] if len(districts) > 0 else "N/A"
    top_district_pct = districts.iloc[0]["pct"] if len(districts) > 0 else 0
    
    top_category = df["category"].value_counts().index[0] if len(df) > 0 else "N/A"
    top_category_pct = df["category"].value_counts().iloc[0] / len(df) * 100
    
    peak_hour = df["hour"].value_counts().index[0] if len(df) > 0 else 0
    peak_day = df["dayofweek"].value_counts().index[0] if len(df) > 0 else "N/A"
    
    high_spikes = weekly[weekly["spike_type"] == "HIGH"]
    worst_spike = high_spikes.loc[high_spikes["pct_change"].idxmax()] if not high_spikes.empty else None
    
    # Dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Key Statistics")
        st.markdown(f"""
        - **Total Incidents:** {total_crimes:,}
        - **Analysis Period:** {date_range_days} days
        - **Average Daily Crimes:** {avg_daily:.1f}
        - **Unique Crime Types:** {df['category'].nunique()}
        - **Police Districts:** {df['pddistrict'].nunique()}
        """)
        
        st.markdown("###  Critical Findings")
        st.markdown(f"""
        - **Highest Crime District:** **{top_district}** ({top_district_pct}% of all crimes)
        - **Most Common Crime:** **{top_category}** ({top_category_pct:.1f}% of incidents)
        - **Peak Crime Window:** **{peak_hour:02d}:00** on **{peak_day}s**
        - **Anomaly Weeks Detected:** **{weekly['is_spike'].sum()}** ({len(high_spikes)} HIGH spikes)
        """)
    
    with col2:
        st.markdown("###  Trend Analysis")
        
        # Add trend gauge
        recent_avg = weekly["crime_count"].tail(4).mean()
        overall_avg = weekly["crime_count"].mean()
        trend_pct = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=trend_pct,
            title={"text": "Recent Trend (4-week vs Overall)"},
            delta={"reference": 0},
            gauge={
                "axis": {"range": [-50, 50]},
                "bar": {"color": "#58a6ff"},
                "steps": [
                    {"range": [-50, -20], "color": "#2d1215"},
                    {"range": [-20, 20], "color": "#12261e"},
                    {"range": [20, 50], "color": "#2d1215"}
                ],
                "threshold": {
                    "line": {"color": "#ff6b6b", "width": 4},
                    "thickness": 0.75,
                    "value": 20
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if worst_spike is not None:
            st.markdown(f"""
            ###  Critical Alert
            **Worst Spike Detected:** Week of {worst_spike['week_start'].strftime('%B %d, %Y')}
            - Increase: **+{worst_spike['pct_change']:.1f}%**
            - Z-Score: **{worst_spike['z_score']:.2f}** (significantly abnormal)
            """)
    
    # Recommendations
    st.markdown("---")
    st.markdown("###  Recommended Actions")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        <div class='alert-red'>
        <b> IMMEDIATE ACTIONS</b><br>
        • Increase patrol presence in <b>{}</b><br>
        • Focus on <b>{}</b> hour window<br>
        • Deploy resources to <b>{}</b> category hotspots
        </div>
        """.format(top_district, f"{peak_hour:02d}:00-{(peak_hour+2)%24:02d}:00", top_category), unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("""
        <div class='alert-blue'>
        <b> MONITORING PROTOCOL</b><br>
        • Track weekly Z-scores (alert at >{})<br>
        • Review top 5 districts weekly<br>
        • Monitor {}/{} crime type trends
        </div>
        """.format(spike_thresh, top_category, top_category), unsafe_allow_html=True)
    
    with rec_col3:
        st.markdown("""
        <div class='alert-green'>
        <b> STRATEGIC PLANNING</b><br>
        • Re-evaluate clusters quarterly<br>
        • Adjust patrol zones based on hotspots<br>
        • Share intelligence with {} district
        </div>
        """.format(top_district), unsafe_allow_html=True)
    
    # Data export
    st.markdown("---")
    st.markdown("###  Export Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        st.download_button(
            " Export Filtered Data",
            df.to_csv(index=False).encode(),
            "CityGuard_filtered_data.csv",
            "text/csv"
        )
    
    with export_col2:
        st.download_button(
            " Export Weekly Summary",
            weekly.to_csv(index=False).encode(),
            "weekly_summary.csv",
            "text/csv"
        )
    
    with export_col3:
        st.download_button(
            " Export District Rankings",
            districts.to_csv(index=False).encode(),
            "district_rankings.csv",
            "text/csv"
        )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#6e7681;font-size:12px;'>
     CityGuard Crime Intelligence Platform | Powered by Streamlit & AI Analytics<br>
    Data Source: SFPD Crime Incident Reporting System
    </div>
    """,
    unsafe_allow_html=True
)