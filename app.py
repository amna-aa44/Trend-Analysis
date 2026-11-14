import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

st.set_page_config(
    page_title="The story of Shakargarh's resources: Crop-Water stress in the past 4 decades",
    page_icon="🌾",
    layout="wide",
)


# --- Utility function for safe trend statistics ---
def safe_mann_kendall(series):
    s = pd.Series(series).dropna()
    if len(s) < 3:
        class EmptyMKResult:
            p = 1.0
            slope = 0.0
            trend = "no trend"

        return EmptyMKResult()

    try:
        return mk.hamed_rao_modification_test(s)
    except Exception:
        class EmptyMKResult:
            p = 1.0
            slope = 0.0
            trend = "no trend"

        return EmptyMKResult()


# --- 1. DATA LOADER & PREPROCESSOR (AUTO-DETECTS NASA GIOVANNI FILENAMES) ---
@st.cache_data
def load_data():
    def read_giovanni(pattern, col_name):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing CSV matching pattern: {pattern}")

        filepath = matches[0]

        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.readlines()

        skip = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith("time") or (len(stripped) >= 4 and stripped[:4].isdigit()):
                skip = i if stripped.lower().startswith("time") else max(0, i - 1)
                break

        df_tmp = pd.read_csv(filepath, skiprows=skip)

        if df_tmp.empty:
            raise ValueError(f"No data found in file: {filepath}")

        if "Date" not in df_tmp.columns:
            df_tmp = df_tmp.iloc[:, :2].copy()
            df_tmp.columns = ["Date", col_name]
        else:
            df_tmp = df_tmp[[df_tmp.columns[0], df_tmp.columns[1]]].copy()
            df_tmp.columns = ["Date", col_name]

        df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce")
        df_tmp[col_name] = pd.to_numeric(df_tmp[col_name], errors="coerce")
        return df_tmp.dropna(subset=["Date", col_name]).reset_index(drop=True)

    try:
        et = read_giovanni("*Evap_tavg*.csv", "ET_raw")
        sm_surf = read_giovanni("*SoilMoi00_10cm*.csv", "SM_0_10cm")
        sm_root = read_giovanni("*SoilMoi10_40cm*.csv", "SM_10_40cm")
    except FileNotFoundError as exc:
        st.error(f"Missing NASA Giovanni CSV files. Please ensure the expected files are in the project folder. Details: {exc}")
        st.stop()

    merged = et.merge(sm_surf, on="Date", how="inner").merge(sm_root, on="Date", how="inner")

    if merged.empty:
        st.error("No overlapping dates were found across the ET and soil moisture data files.")
        st.stop()

    merged["ET_mm_month"] = merged["ET_raw"] * 2592000.0  # Convert kg/m2/s to mm/month
    merged["Aridity_Stress_Idx"] = merged["ET_mm_month"] / (merged["SM_10_40cm"] * 300.0)
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month

    return merged[merged["Year"] <= 2025].copy()


try:
    df_complete = load_data()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()


# --- 2. STORY-DRIVEN HEADER ---
st.title("🌾 The story of Shakargarh's resources: Crop-Water stress in the past 4 decades")
st.markdown(
    """
    **NASA 2026 Space Apps Challenge: *Be An Earth System Trend Detective!***
    *For generations, farming families between the River Ravi and Nullah Bein have timed their sowing and harvesting to the rhythm of the seasons. Using 44 years of NASA satellite-assimilated data, we trace how rising atmospheric demand and changing soil moisture affect crop production in Shakargarh.*
    """
)


# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🧭 Explore Shakargarh's Seasons")
season = st.sidebar.selectbox(
    "1. Choose a Farming Season",
    [
        "Full Agricultural Year (Annual Mean)",
        "Kharif Season (Jun–Sep: Monsoon & Rice)",
        "Rabi Season (Nov–Apr: Winter Wheat)",
    ],
)
split_year = st.sidebar.slider("2. Compare Before vs. After Year", 1995, 2015, 2004)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Why this matters:** Even when annual rainfall or soil moisture rises, faster evaporation under warmer skies can leave crops thirstier during critical dry months—a pattern known as progressive aridification."
)


# --- 4. FILTER & COMPUTE STATISTICAL TRENDS ---
if season == "Full Agricultural Year (Annual Mean)":
    sub_m = df_complete.copy()
    crop_note = (
        "Over the past four decades, the air above Shakargarh has been exerting increasing evaporative demand, while soil moisture and crop stress conditions show persistent long-term change."
    )
elif season == "Kharif Season (Jun–Sep: Monsoon & Rice)":
    sub_m = df_complete[df_complete["Month"].isin([6, 7, 8, 9])].copy()
    crop_note = (
        "During the monsoon season, rainfall and soil moisture patterns are highly important, but crop water demand can still create stress when atmospheric demand outpaces the moisture available to root zones."
    )
else:
    sub_m = df_complete[df_complete["Month"].isin([11, 12, 1, 2, 3, 4])].copy()
    crop_note = (
        "Winter wheat is sensitive to soil moisture carryover and seasonal water balance, so changes in evaporative demand can significantly affect crop productivity during the Rabi season."
    )

if sub_m.empty:
    st.warning("No rows are available for the selected season.")
    st.stop()

ann = sub_m.groupby("Year", as_index=True).mean(numeric_only=True).reset_index()

mk_et = safe_mann_kendall(ann["ET_mm_month"])
mk_sm = safe_mann_kendall(ann["SM_10_40cm"])
mk_ar = safe_mann_kendall(ann["Aridity_Stress_Idx"])

corr_df = sub_m[["SM_10_40cm", "ET_mm_month"]].dropna()
if len(corr_df) >= 2:
    r_val, p_corr = pearsonr(corr_df["SM_10_40cm"], corr_df["ET_mm_month"])
else:
    r_val, p_corr = np.nan, np.nan

if ann.empty:
    st.warning("No annual summary data available for analysis.")
    st.stop()

early = ann[ann["Year"] < split_year]
late = ann[ann["Year"] >= split_year]

# Safe percentage shift calculation
def percent_shift(old_val, new_val):
    if pd.isna(old_val) or abs(old_val) < 1e-12:
        return 0.0
    return ((new_val - old_val) / old_val) * 100.0

et_shift = percent_shift(early["ET_mm_month"].mean() if not early.empty else np.nan, late["ET_mm_month"].mean() if not late.empty else np.nan)
sm_shift = percent_shift(early["SM_10_40cm"].mean() if not early.empty else np.nan, late["SM_10_40cm"].mean() if not late.empty else np.nan)
ar_shift = percent_shift(early["Aridity_Stress_Idx"].mean() if not early.empty else np.nan, late["Aridity_Stress_Idx"].mean() if not late.empty else np.nan)

st.info(f"**What the data tells us ({season}):** {crop_note}")

# --- 5. KEY METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Crop Water Demand (ET)",
    f"{et_shift:+.2f}%",
    f"p = {mk_et.p:.4f} ({'Significant' if mk_et.p < 0.05 else 'Stable / Variable'})",
)
c2.metric(
    "Root-Zone Soil Water (10–40 cm)",
    f"{sm_shift:+.2f}%",
    f"p = {mk_sm.p:.4f} ({'Significant' if mk_sm.p < 0.05 else 'Stable / Variable'})",
)
c3.metric(
    "Crop-Water Stress Ratio",
    f"{ar_shift:+.2f}%",
    f"Slope: {mk_ar.slope:+.5f}/yr",
)
c4.metric(
    "Soil-to-Crop Link (r)",
    f"{r_val:.3f}" if not pd.isna(r_val) else "N/A",
    f"p = {p_corr:.2e}" if not pd.isna(p_corr) else "p = N/A",
)

# --- 6. TABS FOR CHARTS & LOCAL STORY ---
tab1, tab2 = st.tabs(["📈 4-Decade Trends & Water Stress", "🌱 What This Means for Local Farmers"])

with tab1:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"1. Crop & Soil Water Loss (Evapotranspiration, mm/mo) | Change: {mk_et.slope:+.4f}/yr (p = {mk_et.p:.4f})",
            f"2. Root-Zone Soil Moisture (10–40 cm depth, m³/m³) | Change: {mk_sm.slope:+.5f}/yr (p = {mk_sm.p:.4f})",
            f"3. Crop-Water Stress Index (Water Demand vs. Soil Supply) | Change: {mk_ar.slope:+.5f}/yr (p = {mk_ar.p:.4f})",
        ),
    )

    # Panel 1: ET
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=ann["ET_mm_month"],
            mode="lines+markers",
            name="Evapotranspiration",
            line=dict(color="#c84b31", width=2),
        ),
        row=1,
        col=1,
    )
    z_et = np.polyfit(ann["Year"], ann["ET_mm_month"], 1)
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=np.poly1d(z_et)(ann["Year"]),
            mode="lines",
            line=dict(color="#6b1d0e", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Panel 2: Soil Moisture
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=ann["SM_10_40cm"],
            mode="lines+markers",
            name="Root Soil Moisture",
            line=dict(color="#005a9c", width=2),
        ),
        row=2,
        col=1,
    )
    z_sm = np.polyfit(ann["Year"], ann["SM_10_40cm"], 1)
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=np.poly1d(z_sm)(ann["Year"]),
            mode="lines",
            line=dict(color="#002b49", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Panel 3: Crop-Water Stress Index
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=ann["Aridity_Stress_Idx"],
            mode="lines+markers",
            name="Crop-Water Stress Index",
            line=dict(color="#d97706", width=2),
        ),
        row=3,
        col=1,
    )
    z_ar = np.polyfit(ann["Year"], ann["Aridity_Stress_Idx"], 1)
    fig.add_trace(
        go.Scatter(
            x=ann["Year"],
            y=np.poly1d(z_ar)(ann["Year"]),
            mode="lines",
            line=dict(color="#78350f", dash="dash"),
        ),
        row=3,
        col=1,
    )

    fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
    fig.update_layout(height=740, template="plotly_white", showlegend=False, margin=dict(t=45, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Evidence (Autocorrelation-Corrected Mann-Kendall Test)")
    summary_df = pd.DataFrame(
        {
            "Resource Indicator": [
                "Evapotranspiration (Crop & Soil Water Use)",
                "Root-Zone Soil Moisture (10–40 cm)",
                "Crop-Water Stress Index",
            ],
            "44-Year Direction": [
                mk_et.trend.capitalize(),
                mk_sm.trend.capitalize(),
                mk_ar.trend.capitalize(),
            ],
            "Annual Rate of Change": [
                f"{mk_et.slope:+.4f} mm/mo",
                f"{mk_sm.slope:+.5f} m³/m³",
                f"{mk_ar.slope:+.5f} /yr",
            ],
            "p-value": [
                f"{mk_et.p:.4f}",
                f"{mk_sm.p:.4f}",
                f"{mk_ar.p:.4f}",
            ],
            "Statistically Proven (p < 0.05)": [
                "Yes" if mk_et.p < 0.05 else "No (High Year-to-Year Swing)",
                "Yes" if mk_sm.p < 0.05 else "No (High Year-to-Year Swing)",
                "Yes" if mk_ar.p < 0.05 else "No (High Year-to-Year Swing)",
            ],
            f"Change After {split_year}": [
                f"{et_shift:+.2f}%",
                f"{sm_shift:+.2f}%",
                f"{ar_shift:+.2f}%",
            ],
        }
    )
    st.table(summary_df)

with tab2:
    st.markdown(
        """
        ### Reading Between the Lines: Two Seasons, Two Opposite Risks
        1. **Too Much Water at Once in Summer (Kharif):** In seasons with strong monsoon rainfall, soil moisture can rise quickly, but excess water may not translate into better crop growth if the soil becomes saturated or if drainage and timing do not match crop demand.
        2. **Winter Scarcity in Rabi (Dry Season):** Wheat and other winter crops depend on preserved soil moisture and irrigation. If atmospheric demand rises faster than the soil can supply water, crop stress intensifies even when total annual moisture seems sufficient.

        In practical terms, Shakargarh's farmers face a dual challenge: managing flood-like monsoon pulses in summer while protecting crops from persistent evaporative stress during the dry season.
        """
    )

    st.markdown("### Local Interpretation")
    st.write(
        "These signals are relevant to local farmers because they describe how available water is distributed across the year. In hot and dry periods, water demand can exceed what the root zone can provide, reducing crop resilience and raising the risk of lower yields."
    )

    st.caption("Dataset note: This analysis uses NASA Giovanni monthly time series and compares long-term trends before and after the selected year.")


# Prevent Streamlit from re-running the entire app on every interaction in ways that can break file loading
# (This is already managed by caching and the use of sidebar controls.)
"""
This file is intended to run as a Streamlit app.
Run with:
    streamlit run app.py
"""
