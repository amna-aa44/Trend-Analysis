import glob
import streamlit as st
import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Shakargarh Earth System Trend Detective", page_icon="🌍", layout="wide")

# --- 1. DATA LOADER & PREPROCESSOR (AUTO-DETECTS NASA GIOVANNI FILENAMES) ---
@st.cache_data
def load_data():
    def read_giovanni(pattern, col_name):
        matches = glob.glob(pattern)
        if not matches:
            st.error(f"Missing CSV matching pattern: {pattern}")
            st.stop()
        filepath = matches[0]
        with open(filepath, 'r') as f:
            lines = f.readlines()
        skip = 0
        for i, line in enumerate(lines):
            if line.lower().startswith('time') or line[:4].isdigit():
                skip = i if line.lower().startswith('time') else i - 1
                break
        df_tmp = pd.read_csv(filepath, skiprows=max(0, skip))
        df_tmp.columns = ['Date', col_name]
        df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])
        df_tmp[col_name] = pd.to_numeric(df_tmp[col_name], errors='coerce')
        return df_tmp.dropna()

    et = read_giovanni('*Evap_tavg*.csv', 'ET_raw')
    sm_surf = read_giovanni('*SoilMoi00_10cm*.csv', 'SM_0_10cm')
    sm_root = read_giovanni('*SoilMoi10_40cm*.csv', 'SM_10_40cm')

    merged = et.merge(sm_surf, on='Date').merge(sm_root, on='Date')
    merged['ET_mm_month'] = merged['ET_raw'] * 2592000
    merged['Year'] = merged['Date'].dt.year
    merged['Month'] = merged['Date'].dt.month
    return merged[merged['Year'] <= 2025]

df_complete = load_data()

# --- 2. HEADER & METHODOLOGY ---
st.title("🌍 Earth System Trend Detective: Shakargarh Hydro-Agricultural Observatory")
st.markdown("""
**NASA 2026 Space Apps Challenge** | Inspired by Prof. Matthew McCabe's Multi-Sensor Agricultural Hydrology Framework  
*Analyzing 44 years (1982–2025, N = 528 complete months) of NASA FLDAS root-zone soil moisture and evapotranspiration across the Ravi–Bein agricultural corridor (32.0°N–32.5°N, 75.0°E–75.5°E).*
""")

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Observatory Controls")
season = st.sidebar.selectbox(
    "1. Select Cropping / Hydrological Regime",
    [
        "Annual Mean (Full Year)",
        "Kharif Season (Jun–Sep: Rice & Monsoon)",
        "Rabi Season (Nov–Apr: Wheat)"
    ]
)
split_year = st.sidebar.slider("2. Regime-Shift Split Year (Before vs. After)", 1995, 2015, 2004)

# --- 4. FILTER & COMPUTE STATISTICS ---
if season == "Annual Mean (Full Year)":
    sub_m = df_complete.copy()
    crop_note = "Full-year hydrological balance across Tehsil Shakargarh (Ravi–Bein agricultural corridor)."
elif season == "Kharif Season (Jun–Sep: Rice & Monsoon)":
    sub_m = df_complete[df_complete['Month'].isin([6, 7, 8, 9])]
    crop_note = "Kharif Monsoon Regime: High rainfall & paddy inundation. Rising soil moisture without proportional ET growth signals waterlogging/energy-limited saturation."
else:
    sub_m = df_complete[df_complete['Month'].isin([11, 12, 1, 2, 3, 4])]
    crop_note = "Rabi Winter Regime: Wheat cultivation. Crop transpiration relies heavily on residual root-zone moisture and groundwater buffering."

ann = sub_m.groupby('Year').mean(numeric_only=True).reset_index()

mk_et = mk.hamed_rao_modification_test(ann['ET_mm_month'])
mk_sm = mk.hamed_rao_modification_test(ann['SM_10_40cm'])
r_val, p_corr = pearsonr(sub_m['SM_10_40cm'], sub_m['ET_mm_month'])

early = ann[ann['Year'] < split_year]
late = ann[ann['Year'] >= split_year]
et_shift = ((late['ET_mm_month'].mean() - early['ET_mm_month'].mean()) / early['ET_mm_month'].mean()) * 100
sm_shift = ((late['SM_10_40cm'].mean() - early['SM_10_40cm'].mean()) / early['SM_10_40cm'].mean()) * 100

st.info(f"**Diagnostic Context ({season}):** {crop_note}")

# --- 5. KPI METRICS ---
c1, c2, c3 = st.columns(3)
c1.metric(
    "Root-Zone Soil Moisture Shift",
    f"{sm_shift:+.2f}%",
    f"p = {mk_sm.p:.4f} ({'Significant' if mk_sm.p < 0.05 else 'Not Sig.'})"
)
c2.metric(
    "Evapotranspiration (ET) Shift",
    f"{et_shift:+.2f}%",
    f"p = {mk_et.p:.4f} ({'Significant' if mk_et.p < 0.05 else 'Not Sig.'})"
)
c3.metric(
    "Soil Moisture–ET Coupling (r)",
    f"{r_val:.3f}",
    f"p = {p_corr:.2e}"
)

# --- 6. INTERACTIVE PLOTLY CHART ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
    subplot_titles=(
        f"Root-Zone Soil Moisture (10–40 cm) | Sen's Slope: {mk_sm.slope:+.5f}/yr (p = {mk_sm.p:.4f})",
        f"Agricultural Evapotranspiration (mm/mo) | Sen's Slope: {mk_et.slope:+.4f}/yr (p = {mk_et.p:.4f})"
    )
)

fig.add_trace(go.Scatter(
    x=ann['Year'], y=ann['SM_10_40cm'], mode='lines+markers',
    name='Root Soil Moisture', line=dict(color='#005a9c', width=2)
), row=1, col=1)

z_sm = np.polyfit(ann['Year'], ann['SM_10_40cm'], 1)
fig.add_trace(go.Scatter(
    x=ann['Year'], y=np.poly1d(z_sm)(ann['Year']), mode='lines',
    name='SM Trendline', line=dict(color='#002b49', dash='dash')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=ann['Year'], y=ann['ET_mm_month'], mode='lines+markers',
    name='Evapotranspiration', line=dict(color='#c84b31', width=2)
), row=2, col=1)

z_et = np.polyfit(ann['Year'], ann['ET_mm_month'], 1)
fig.add_trace(go.Scatter(
    x=ann['Year'], y=np.poly1d(z_et)(ann['Year']), mode='lines',
    name='ET Trendline', line=dict(color='#6b1d0e', dash='dash')
), row=2, col=1)

fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
fig.update_layout(height=550, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- 7. STATISTICAL SUMMARY TABLE ---
st.subheader("Hamed–Rao Autocorrelation-Corrected Mann-Kendall Summary")
summary_df = pd.DataFrame({
    "Variable": ["Root-Zone Soil Moisture (10–40 cm)", "Evapotranspiration (ET)"],
    "44-Yr Trend": [mk_sm.trend.capitalize(), mk_et.trend.capitalize()],
    "Sen's Slope (/yr)": [f"{mk_sm.slope:+.5f} m³/m³", f"{mk_et.slope:+.4f} mm/mo"],
    "p-value": [f"{mk_sm.p:.4f}", f"{mk_et.p:.4f}"],
    "Statistical Significance (α = 0.05)": [
        "Significant" if mk_sm.p < 0.05 else "Not Significant",
        "Significant" if mk_et.p < 0.05 else "Not Significant"
    ],
    f"Pre- vs. Post-{split_year} Change": [f"{sm_shift:+.2f}%", f"{et_shift:+.2f}%"]
})
st.table(summary_df)
