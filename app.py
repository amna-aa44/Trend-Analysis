import glob
import streamlit as st
import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="The story of Shakargarh's resources: Crop-Water stress in the past 4 decades",
    page_icon="🌾",
    layout="wide"
)

# --- 1. DATA LOADER & PREPROCESSOR (AUTO-DETECTS NASA GIOVANNI FILENAMES) ---
@st.cache_data
def load_data():
    def read_giovanni(pattern, col_name):
        matches = glob.glob(pattern)
        if not matches:
            st.error(f"Missing CSV matching pattern: {pattern}")
            st.stop()
        filepath = matches
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
    merged['ET_mm_month'] = merged['ET_raw'] * 2592000  # Convert kg/m2/s to mm/month
    # Evaporative Extraction Stress Index: Atmospheric water pull relative to root-zone moisture
    merged['Aridity_Stress_Idx'] = merged['ET_mm_month'] / (merged['SM_10_40cm'] * 300.0)
    merged['Year'] = merged['Date'].dt.year
    merged['Month'] = merged['Date'].dt.month
    return merged[merged['Year'] <= 2025]

df_complete = load_data()

# --- 2. STORY-DRIVEN HEADER ---
st.title("🌾 The story of Shakargarh's resources: Crop-Water stress in the past 4 decades")
st.markdown("""
**NASA 2026 Space Apps Challenge: *Be An Earth System Trend Detective!***  
*For generations, farming families between the River Ravi and Nullah Bein have timed their sowing and harvesting to the rhythm of the seasons. Using 44 years of NASA satellite-assimilated data (`1982–2025`, `N = 528 months`), this observatory uncovers how warming temperatures, shifting monsoons, and rising crop water demand have quietly reshaped Shakargarh’s fields.*
""")

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🧭 Explore Shakargarh's Seasons")
season = st.sidebar.selectbox(
    "1. Choose a Farming Season",
    [
        "Full Agricultural Year (Annual Mean)",
        "Kharif Season (Jun–Sep: Monsoon & Rice)",
        "Rabi Season (Nov–Apr: Winter Wheat)"
    ]
)
split_year = st.sidebar.slider("2. Compare Before vs. After Year", 1995, 2015, 2004)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Why this matters:** Even when annual rainfall or soil moisture rises, faster evaporation under warmer skies "
    "can leave crops thirstier during critical dry months—a pattern known as progressive aridification."
)

# --- 4. FILTER & COMPUTE STATISTICAL TRENDS ---
if season == "Full Agricultural Year (Annual Mean)":
    sub_m = df_complete.copy()
    crop_note = (
        "Over the past four decades, the air above Shakargarh is pulling water out of crops and soil 18.52% faster (p ≈ 0.01) "
        "than it did in the 1980s and 1990s—slightly outpacing the 17.95% rise in root-zone soil moisture (p ≈ 0.04)."
    )
elif season == "Kharif Season (Jun–Sep: Monsoon & Rice)":
    sub_m = df_complete[df_complete['Month'].isin()]
    crop_note = (
        "During the summer monsoon, root-zone soil moisture has climbed significantly (p < 0.05) due to heavier downpours, "
        "yet crop evapotranspiration has leveled off—showing that excess water often floods or waterlogs paddy fields rather than boosting growth."
    )
else:
    sub_m = df_complete[df_complete['Month'].isin()]
    crop_note = (
        "Winter wheat tells a quieter, more fragile story: soil moisture shows no statistically significant gain (p > 0.05) over 44 years, "
        "meaning Rabi farmers depend heavily on leftover monsoon moisture and tube-well irrigation to protect yields."
    )

ann = sub_m.groupby('Year').mean(numeric_only=True).reset_index()

mk_et = mk.hamed_rao_modification_test(ann['ET_mm_month'])
mk_sm = mk.hamed_rao_modification_test(ann['SM_10_40cm'])
mk_ar = mk.hamed_rao_modification_test(ann['Aridity_Stress_Idx'])
r_val, p_corr = pearsonr(sub_m['SM_10_40cm'], sub_m['ET_mm_month'])

early = ann[ann['Year'] < split_year]
late = ann[ann['Year'] >= split_year]
et_shift = ((late['ET_mm_month'].mean() - early['ET_mm_month'].mean()) / early['ET_mm_month'].mean()) * 100
sm_shift = ((late['SM_10_40cm'].mean() - early['SM_10_40cm'].mean()) / early['SM_10_40cm'].mean()) * 100
ar_shift = ((late['Aridity_Stress_Idx'].mean() - early['Aridity_Stress_Idx'].mean()) / early['Aridity_Stress_Idx'].mean()) * 100

st.info(f"**What the data tells us ({season}):** {crop_note}")

# --- 5. KEY METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Crop Water Demand (ET)",
    f"{et_shift:+.2f}%",
    f"p = {mk_et.p:.4f} ({'Significant' if mk_et.p < 0.05 else 'Stable / Variable'})"
)
c2.metric(
    "Root-Zone Soil Water (10–40 cm)",
    f"{sm_shift:+.2f}%",
    f"p = {mk_sm.p:.4f} ({'Significant' if mk_sm.p < 0.05 else 'Stable / Variable'})"
)
c3.metric(
    "Crop-Water Stress Ratio",
    f"{ar_shift:+.2f}%",
    f"Slope: {mk_ar.slope:+.5f}/yr"
)
c4.metric(
    "Soil-to-Crop Link (r)",
    f"{r_val:.3f}",
    f"p = {p_corr:.2e}"
)

# --- 6. TABS FOR CHARTS & LOCAL STORY ---
tab1, tab2 = st.tabs(["📈 4-Decade Trends & Water Stress", "🌱 What This Means for Local Farmers"])

with tab1:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            f"1. Crop & Soil Water Loss (Evapotranspiration, mm/mo) | Change: {mk_et.slope:+.4f}/yr (p = {mk_et.p:.4f})",
            f"2. Root-Zone Soil Moisture (10–40 cm depth, m³/m³) | Change: {mk_sm.slope:+.5f}/yr (p = {mk_sm.p:.4f})",
            f"3. Crop-Water Stress Index (Water Demand vs. Soil Supply) | Change: {mk_ar.slope:+.5f}/yr (p = {mk_ar.p:.4f})"
        )
    )

    # Panel 1: ET
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=ann['ET_mm_month'], mode='lines+markers',
        name='Evapotranspiration', line=dict(color='#c84b31', width=2)
    ), row=1, col=1)
    z_et = np.polyfit(ann['Year'], ann['ET_mm_month'], 1)
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=np.poly1d(z_et)(ann['Year']), mode='lines',
        line=dict(color='#6b1d0e', dash='dash')
    ), row=1, col=1)

    # Panel 2: Soil Moisture
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=ann['SM_10_40cm'], mode='lines+markers',
        name='Root Soil Moisture', line=dict(color='#005a9c', width=2)
    ), row=2, col=1)
    z_sm = np.polyfit(ann['Year'], ann['SM_10_40cm'], 1)
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=np.poly1d(z_sm)(ann['Year']), mode='lines',
        line=dict(color='#002b49', dash='dash')
    ), row=2, col=1)

    # Panel 3: Crop-Water Stress Index
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=ann['Aridity_Stress_Idx'], mode='lines+markers',
        name='Crop-Water Stress Index', line=dict(color='#d97706', width=2)
    ), row=3, col=1)
    z_ar = np.polyfit(ann['Year'], ann['Aridity_Stress_Idx'], 1)
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=np.poly1d(z_ar)(ann['Year']), mode='lines',
        line=dict(color='#78350f', dash='dash')
    ), row=3, col=1)

    fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
    fig.update_layout(height=740, template="plotly_white", showlegend=False, margin=dict(t=45, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Evidence (Autocorrelation-Corrected Mann-Kendall Test)")
    summary_df = pd.DataFrame({
        "Resource Indicator": [
            "Evapotranspiration (Crop & Soil Water Use)",
            "Root-Zone Soil Moisture (10–40 cm)",
            "Crop-Water Stress Index"
        ],
        "44-Year Direction": [
            mk_et.trend.capitalize(),
            mk_sm.trend.capitalize(),
            mk_ar.trend.capitalize()
        ],
        "Annual Rate of Change": [
            f"{mk_et.slope:+.4f} mm/mo",
            f"{mk_sm.slope:+.5f} m³/m³",
            f"{mk_ar.slope:+.5f} /yr"
        ],
        "p-value": [
            f"{mk_et.p:.4f}",
            f"{mk_sm.p:.4f}",
            f"{mk_ar.p:.4f}"
        ],
        "Statistically Proven (p < 0.05)": [
            "Yes" if mk_et.p < 0.05 else "No (High Year-to-Year Swing)",
            "Yes" if mk_sm.p < 0.05 else "No (High Year-to-Year Swing)",
            "Yes" if mk_ar.p < 0.05 else "No (High Year-to-Year Swing)"
        ],
        f"Change After {split_year}": [
            f"{et_shift:+.2f}%",
            f"{sm_shift:+.2f}%",
            f"{ar_shift:+.2f}%"
        ]
    })
    st.table(summary_df)

with tab2:
    st.markdown("""
    ### Reading Between the Lines: Two Seasons, Two Opposite Risks
    1. **Too Much Water at Once in Summer (Kharif):** Since the early 2000s, monsoon soil moisture in Shakargarh has risen significantly ($p < 0.05$), yet crop water use (ET) has not increased alongside it. For farmers along **Nullah Bein** and the **River Ravi**, this reflects sudden, heavy monsoon bursts that saturate and waterlog fields rather than providing steady, usable moisture.
