# -*- coding: utf-8 -*-
"""
Rocket Launch Availability Dashboard & FDI Engine
Configured for GitHub / Local File Caching (Zero API calls when data exists in /data)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Detect Pyodide / Browser environment
IS_PYODIDE = "pyodide" in sys.modules

if not IS_PYODIDE:
    import requests

# =============================================================================
# 0. STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Rocket Launch Window Analyzer",
    page_icon="🚀",
    layout="wide"
)

# =============================================================================
# 1. SOUTH AFRICAN FIRE DANGER INDEX (FDI) ENGINE
# =============================================================================
def compute_sa_fdi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes South African Fire Danger Index (FDI) matching exact reference logic."""
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # 1. Base FDI Calculation (FDI1)
    df['FDI1'] = (df['Temp'] - 35) - ((35 - df['Temp']) / 30) + (0.37 * (100 - df['RH'])) + 30

    # 2. Wind Correction Factor (FDI2) - Wind Speed in m/s
    ws = df['WS']
    wind_add = np.select(
        [
            (ws >= 0) & (ws < (3 / 3.6)),
            (ws >= (3 / 3.6)) & (ws < (9 / 3.6)),
            (ws >= (9 / 3.6)) & (ws < (17 / 3.6)),
            (ws >= (17 / 3.6)) & (ws < (26 / 3.6)),
            (ws >= (26 / 3.6)) & (ws < (33 / 3.6)),
            (ws >= (33 / 3.6)) & (ws < (37 / 3.6)),
            (ws >= (37 / 3.6)) & (ws < (42 / 3.6)),
            (ws >= (42 / 3.6)) & (ws < (46 / 3.6)),
            ws >= (46 / 3.6)
        ],
        [0, 5, 10, 15, 20, 25, 30, 35, 40],
        default=0
    )
    df['FDI2'] = df['FDI1'] + wind_add

    # 3. Track Most Recent Prior Rainfall & Days Elapsed
    last_rain_ts = df['Timestamp'].where(df['Rain'] > 0).shift(1).ffill()
    last_rain_amt = df['Rain'].where(df['Rain'] > 0).shift(1).ffill().fillna(0)

    df['Days_Since_Rainfall'] = (df['Timestamp'] - last_rain_ts).dt.days.fillna(21).astype(int)
    df['Rainfall_Amount'] = last_rain_amt

    # 4. Multiplicative Rain Decay Factor Matrix
    factor = np.ones(len(df))
    r_amt = df['Rainfall_Amount']
    dsr = df['Days_Since_Rainfall']

    # Band 1: [0.0001, 2.7)
    m = (r_amt >= 0.0001) & (r_amt < 2.7)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.7
    factor[m & (dsr == 2)] = 0.9

    # Band 2: [2.7, 5.3)
    m = (r_amt >= 2.7) & (r_amt < 5.3)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.6
    factor[m & (dsr == 2)] = 0.8
    factor[m & (dsr == 3)] = 0.9

    # Band 3: [5.3, 7.7)
    m = (r_amt >= 5.3) & (r_amt < 7.7)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.5
    factor[m & (dsr == 2)] = 0.7
    factor[m & ((dsr == 3) | (dsr == 4))] = 0.9

    # Band 4: [7.7, 10.3)
    m = (r_amt >= 7.7) & (r_amt < 10.3)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.4
    factor[m & (dsr == 2)] = 0.6
    factor[m & (dsr == 3)] = 0.8
    factor[m & ((dsr == 4) | (dsr == 5))] = 0.9

    # Band 5: [10.3, 12.9)
    m = (r_amt >= 10.3) & (r_amt < 12.9)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.4
    factor[m & (dsr == 2)] = 0.6
    factor[m & (dsr == 3)] = 0.7
    factor[m & (dsr == 4)] = 0.8
    factor[m & ((dsr == 5) | (dsr == 6))] = 0.9

    # Band 6: [12.9, 15.4)
    m = (r_amt >= 12.9) & (r_amt < 15.4)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.3
    factor[m & (dsr == 2)] = 0.5
    factor[m & (dsr == 3)] = 0.7
    factor[m & ((dsr == 4) | (dsr == 5))] = 0.8
    factor[m & (dsr == 6)] = 0.9

    # Band 7: [15.4, 20.6)
    m = (r_amt >= 15.4) & (r_amt < 20.6)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.2
    factor[m & (dsr == 2)] = 0.5
    factor[m & (dsr == 3)] = 0.6
    factor[m & (dsr == 4)] = 0.7
    factor[m & ((dsr == 5) | (dsr == 6))] = 0.8
    factor[m & ((dsr == 7) | (dsr == 8))] = 0.9

    # Band 8: [20.6, 25.6)
    m = (r_amt >= 20.6) & (r_amt < 25.6)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.2
    factor[m & (dsr == 2)] = 0.4
    factor[m & (dsr == 3)] = 0.5
    factor[m & ((dsr == 4) | (dsr == 5))] = 0.7
    factor[m & (dsr == 6)] = 0.8
    factor[m & ((dsr == 7) | (dsr == 8))] = 0.9

    # Band 9: [25.6, 38.5)
    m = (r_amt >= 25.6) & (r_amt < 38.5)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.1
    factor[m & (dsr == 2)] = 0.3
    factor[m & (dsr == 3)] = 0.4
    factor[m & ((dsr == 4) | (dsr == 5))] = 0.6
    factor[m & (dsr == 6)] = 0.7
    factor[m & ((dsr == 7) | (dsr == 8))] = 0.8
    factor[m & ((dsr == 9) | (dsr == 10))] = 0.9

    # Band 10: [38.5, 51.2)
    m = (r_amt >= 38.5) & (r_amt < 51.2)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.1
    factor[m & (dsr == 2)] = 0.2
    factor[m & (dsr == 3)] = 0.3
    factor[m & (dsr == 4)] = 0.4
    factor[m & (dsr == 5)] = 0.5
    factor[m & (dsr == 6)] = 0.6
    factor[m & ((dsr == 7) | (dsr == 8))] = 0.7
    factor[m & ((dsr == 9) | (dsr == 10))] = 0.8
    factor[m & ((dsr == 11) | (dsr == 12))] = 0.9

    # Band 11: [51.2, 63.9)
    m = (r_amt >= 51.2) & (r_amt < 63.9)
    factor[m & ((dsr == 0) | (dsr == 1))] = 0.1
    factor[m & (dsr == 2)] = 0.2
    factor[m & (dsr == 3)] = 0.3
    factor[m & (dsr == 4)] = 0.4
    factor[m & (dsr == 5)] = 0.5
    factor[m & (dsr == 6)] = 0.6
    factor[m & ((dsr >= 7) & (dsr <= 10))] = 0.7
    factor[m & ((dsr == 11) | (dsr == 12))] = 0.8
    factor[m & ((dsr >= 13) & (dsr <= 15))] = 0.9

    # Band 12: [63.9, 76.6)
    m = (r_amt >= 63.9) & (r_amt < 76.6)
    factor[m & ((dsr >= 0) & (dsr <= 2))] = 0.1
    factor[m & (dsr == 3)] = 0.2
    factor[m & (dsr == 4)] = 0.3
    factor[m & (dsr == 5)] = 0.4
    factor[m & (dsr == 6)] = 0.5
    factor[m & ((dsr == 7) | (dsr == 8))] = 0.6
    factor[m & ((dsr == 9) | (dsr == 10))] = 0.7
    factor[m & ((dsr >= 11) & (dsr <= 15))] = 0.8
    factor[m & ((dsr >= 16) & (dsr <= 20))] = 0.9

    # Band 13: >= 76.6
    m = r_amt >= 76.6
    factor[m & ((dsr >= 0) & (dsr <= 3))] = 0.1
    factor[m & (dsr == 4)] = 0.2
    factor[m & (dsr == 5)] = 0.4
    factor[m & ((dsr >= 6) & (dsr <= 10))] = 0.6
    factor[m & ((dsr == 11) | (dsr == 12))] = 0.7
    factor[m & ((dsr >= 13) & (dsr <= 15))] = 0.8
    factor[m & ((dsr >= 16) & (dsr <= 20))] = 0.9

    df['FDI'] = df['FDI2'] * factor
    return df

# =============================================================================
# 2. DATA MANAGEMENT (LOCAL REPO CACHE -> API FALLBACK)
# =============================================================================
def fetch_from_open_meteo(lat: float, lon: float, start_year: int, end_year: int) -> tuple[pd.DataFrame, int]:
    """Fetches data from Open-Meteo API when local file is missing."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_year}-01-01&end_date={end_year}-12-31&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover&"
        f"wind_speed_unit=ms&timezone=auto"
    )
    
    if IS_PYODIDE:
        from pyodide.http import open_url  # type: ignore
        import json
        response = open_url(url)
        raw_json = json.loads(response.read())
    else:
        res = requests.get(url, timeout=30)
        if res.status_code != 200:
            st.error(f"Open-Meteo API Error: {res.status_code}")
            return pd.DataFrame(), 0
        raw_json = res.json()

    utc_offset_seconds = raw_json.get("utc_offset_seconds", 0)
    utc_offset_hours = int(round(utc_offset_seconds / 3600))

    data = raw_json["hourly"]
    local_ts = pd.to_datetime(data["time"])
    utc_ts = local_ts - pd.Timedelta(seconds=utc_offset_seconds)

    df = pd.DataFrame({
        "Timestamp": utc_ts,
        "Temp": data["temperature_2m"],
        "RH": data["relative_humidity_2m"],
        "Rain": data["precipitation"],
        "WS": data["wind_speed_10m"],
        "Cloud": data["cloud_cover"]
    })
    
    df = compute_sa_fdi(df)
    df["utc_offset_hours"] = utc_offset_hours
    return df, utc_offset_hours


@st.cache_data(ttl=86400, show_spinner="Loading Weather Data...")
def get_site_data(site_name: str, lat: float, lon: float, start_year: int, end_year: int) -> tuple[pd.DataFrame, int, bool]:
    """
    1. Checks if data file exists in local `./data/` folder.
    2. If found, reads Parquet file directly (0 API calls).
    3. If missing, calls Open-Meteo API and returns flag to allow user to save it.
    """
    os.makedirs("data", exist_ok=True)
    
    # Generate clean filename e.g., data/Arniston_OTR_South_Africa_2020_2025.parquet
    clean_name = site_name.replace(" ", "_").replace("(", "").replace(")", "")
    file_path = os.path.join("data", f"{clean_name}_{start_year}_{end_year}.parquet")

    # Check Repository Cache
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        utc_offset = int(df["utc_offset_hours"].iloc[0]) if "utc_offset_hours" in df.columns else 0
        return df, utc_offset, True  # Loaded from Repo

    # Fallback to API
    df, utc_offset = fetch_from_open_meteo(lat, lon, start_year, end_year)
    return df, utc_offset, False  # Loaded from API

# =============================================================================
# 3. STREAMLIT DASHBOARD INTERFACE
# =============================================================================
def main():
    SITES = {
        "Arniston OTR (South Africa)": {"lat": -34.6674, "lon": 20.2309},
        "Richards Bay (South Africa)": {"lat": -28.783, "lon": 32.0377},
        "Saldanha (South Africa)": {"lat": -33.0117, "lon": 17.9442},
        "Verneukpan (South Africa)": {"lat": -30.1333, "lon": 21.0667},
        "Kiruna Esrange (Sweden)": {"lat": 67.8557, "lon": 20.2251},
        "Wallops Island (USA)": {"lat": 37.8532, "lon": -75.4741}
    }

    # Sidebar Controls
    st.sidebar.title("🚀 Configuration")
    selected_site = st.sidebar.selectbox("Select Launch Site", list(SITES.keys()))
    site_coords = SITES[selected_site]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Analysis Date Range")
    
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_year = st.sidebar.number_input("Start Year", min_value=2000, max_value=2025, value=2020, step=1)
    with col_end:
        end_year = st.sidebar.number_input("End Year", min_value=2000, max_value=2025, value=2025, step=1)

    if start_year > end_year:
        st.sidebar.error("Error: Start Year cannot be greater than End Year.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Launch Commit Criteria (LCC)")

    max_ws = st.sidebar.slider("Max Wind Speed (m/s)", 3.0, 20.0, 10.0, step=0.5)
    max_cloud = st.sidebar.slider("Max Cloud Cover (%)", 10, 100, 50, step=5)
    max_precip = st.sidebar.number_input("Max Rain (mm/hr)", 0.0, 10.0, 0.0, step=0.2)

    apply_fdi = st.sidebar.checkbox("Apply FDI Limit (South Africa)", value=True)
    max_fdi = st.sidebar.slider("Max Fire Danger Index (FDI)", 10.0, 100.0, 46.0, step=1.0, disabled=not apply_fdi)

    # Main Body
    st.title("🚀 Launch Availability Dashboard")
    st.caption("Historical availability analysis, weather data provided by Open-Meteo under CC BY 4.0.")

    # Load Data (Repo -> API Fallback)
    df, utc_offset_hours, loaded_from_repo = get_site_data(
        selected_site, site_coords["lat"], site_coords["lon"], start_year, end_year
    )

    # Status Notification Banner
    clean_name = selected_site.replace(" ", "_").replace("(", "").replace(")", "")
    target_filename = f"{clean_name}_{start_year}_{end_year}.parquet"

    if loaded_from_repo:
        st.success(f"📁 Loaded from repository (`/data/{target_filename}`) — 0 API calls used.")
    else:
        st.warning(f"⚡ File `/data/{target_filename}` not found in repository. Data fetched live via Open-Meteo API.")
        
        # Helper to let user download file to add to /data folder in GitHub
        if not df.empty:
            parquet_bytes = df.to_parquet()
            st.download_button(
                label=f"📥 Download `{target_filename}` for GitHub `/data` folder",
                data=parquet_bytes,
                file_name=target_filename,
                mime="application/octet-stream"
            )

    if not df.empty:
        # Prepare Analysis Columns
        df["month"] = df["Timestamp"].dt.month
        df["hour"] = df["Timestamp"].dt.hour

        # Apply LCC Condition
        condition = (
            (df['Rain'] <= max_precip) &
            (df['WS'] < max_ws) &
            (df['Cloud'] < max_cloud)
        )
        
        if apply_fdi:
            condition &= (df['FDI'] < max_fdi)
            
        df['is_favorable'] = condition.astype(int)

        # Compute Metrics
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        overall_availability = df['is_favorable'].mean() * 100.0

        # Best Hours Calculation (UTC & Local)
        best_hour_utc = int(df.groupby("hour")["is_favorable"].mean().idxmax())
        best_hour_local = int((best_hour_utc + utc_offset_hours) % 24)

        # KPI Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Availability", f"{overall_availability:.2f}%")
        
        best_month_idx = df.groupby("month")["is_favorable"].mean().idxmax()
        col2.metric("Best Month", month_labels[best_month_idx - 1])
        
        col3.metric("Best Launch Hour", f"{best_hour_utc:02d}:00 UTC ({best_hour_local:02d}:00 Local)")

        st.markdown("---")

        # =====================================================================
        # Visualizations (Row 1: Heatmap & Ranking Bar)
        # =====================================================================
        col_map, col_rank = st.columns([2, 1])

        # 1. Heatmap Matrix
        heatmap_matrix = df.groupby(["month", "hour"])["is_favorable"].mean().unstack() * 100.0
        heatmap_matrix.index = [month_labels[m - 1] for m in heatmap_matrix.index]

        fig_map = px.imshow(
            heatmap_matrix,
            labels=dict(x="Hour of Day (UTC)", y="Month", color="% Favorable"),
            x=[f"{h:02d}:00" for h in range(24)],
            y=heatmap_matrix.index,
            color_continuous_scale="Jet",
            zmin=0, zmax=100,
            aspect="auto"
        )

        fig_map.update_xaxes(title_font=dict(size=16), tickfont=dict(size=13))
        fig_map.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))
        fig_map.update_coloraxes(colorbar_title_font=dict(size=14), colorbar_tickfont=dict(size=12))
        fig_map.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))

        with col_map:
            st.subheader("Monthly vs. Hourly Availability Matrix (UTC)")
            st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")

        # 2. Monthly Ranking Bar Chart
        monthly_ranking = (df.groupby("month")["is_favorable"].mean() * 100.0).reset_index()
        monthly_ranking["Month"] = monthly_ranking["month"].apply(lambda m: month_labels[m - 1])
        monthly_ranking = monthly_ranking.sort_values(by="is_favorable", ascending=True)

        fig_bar = px.bar(
            monthly_ranking,
            x="is_favorable",
            y="Month",
            orientation="h",
            labels={"is_favorable": "% Availability", "Month": ""},
            color="is_favorable",
            color_continuous_scale="Jet"
        )

        fig_bar.update_xaxes(title_font=dict(size=16), tickfont=dict(size=13))
        fig_bar.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))
        fig_bar.update_layout(
            height=420, 
            showlegend=False, 
            coloraxis_showscale=False, 
            margin=dict(l=10, r=10, t=30, b=20)
        )

        with col_rank:
            st.subheader("Average Monthly Ranking")
            st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

        # =====================================================================
        # Visualizations (Row 2: Timeline Trend below Heatmap & Ranking)
        # =====================================================================
        st.markdown("---")
        st.subheader("📈 Monthly Availability Timeline")

        # Group by Month-Start and count total hours per bucket
        monthly_trend = (
            df.groupby(pd.Grouper(key="Timestamp", freq="MS"))
            .agg(
                availability_pct=("is_favorable", lambda x: x.mean() * 100.0),
                hour_count=("is_favorable", "count")
            )
            .reset_index()
        )

        # Filter out partial edge months (< 500 hours) caused by UTC offset shifts
        monthly_trend = monthly_trend[monthly_trend["hour_count"] >= 500]

        fig_line = px.line(
            monthly_trend,
            x="Timestamp",
            y="availability_pct",
            labels={"Timestamp": "", "availability_pct": "% Availability"}
        )

        fig_line.update_traces(
            name=selected_site,
            showlegend=True,
            line=dict(width=2.5, color="mediumspringgreen")
        )

        fig_line.update_yaxes(
            range=[0, 100],
            title_font=dict(size=16),
            tickfont=dict(size=13),
            tickformat=".1f"
        )

        fig_line.update_xaxes(
            title_font=dict(size=16),
            tickfont=dict(size=13),
            dtick="M12",
            tickformat="%Y"
        )

        fig_line.update_layout(
            height=380,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                title="",
                font=dict(size=14)
            ),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")

# =============================================================================
# 4. EXECUTION SWITCH
# =============================================================================
if __name__ == "__main__":
    if IS_PYODIDE:
        main()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                main()
            else:
                print("To run locally in Spyder or Terminal, use:")
                print("streamlit run streamlit_app.py")
        except ImportError:
            main()