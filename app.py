import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

st.set_page_config(page_title="The story of Shakargarh's resources: Crop-Water stress in the past 4 decades", page_icon="🌾", layout="wide")


def safe_mk(series):
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return type("MK", (), {"p": 1.0, "slope": 0.0, "trend": "no trend"})()
    try:
        return mk.hamed_rao_modification_test(s)
    except Exception:
        return type("MK", (), {"p": 1.0, "slope": 0.0, "trend": "no trend"})()


@st.cache_data
def load_data():
    def read_giovanni(pattern, col_name):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing CSV matching: {pattern}")
        filepath = matches[0]
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.readlines()
        skip = 0
        for i, line in enumerate(lines):
            st_line = line.strip()
            if st_line.lower().startswith("time") or (len(st_line) >= 4 and st_line[:4].isdigit()):
                skip = i if st_line.lower().startswith("time") else max(0, i - 1)
                break
        df_tmp = pd.read_csv(filepath, skiprows=skip).iloc[:, :2].copy()
        df_tmp.columns = ["Date", col_name]
        df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce")
        df_tmp[col_name] = pd.to_numeric(df_tmp[col_name], errors="coerce")
        return df_tmp.dropna().reset_index(drop=True)

    et = read_giovanni("*Evap_tavg*.csv", "ET_raw")
    sm_surf = read_giovanni("*SoilMoi00_10cm*.csv", "SM_0_10cm")
    sm_root = read_giovanni("*SoilMoi10_40cm*.csv", "SM_10_40cm")

    merged = et.merge(sm_surf, on="Date").merge(sm_root, on="Date")
    merged["ET_mm_month"] = merged["ET_raw"] * 2592000.0
    merged["Aridity_Stress_Idx"] = merged["ET_mm_month"] / (merged["SM_10_40cm"] * 300.0)
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month

    ndvi_matches = sorted(glob.glob("*MOD13A3*.csv"))
    ndvi_df = None
    if ndvi_matches:
        raw_ndvi = pd.read_csv(ndvi_matches[0], encoding="utf-8-sig")
        ndvi_cols = [c for c in raw_ndvi.columns if "NDVI" in c and "Quality" not in c and "VI_" not in c]
        if ndvi_cols and "Date" in raw_ndvi.columns:
            raw_ndvi["Date"] = pd.to_datetime(raw_ndvi["Date"], errors="coerce")
            raw_ndvi["NDVI"] = pd.to_numeric(raw_ndvi[ndvi_cols[0]], errors="coerce")
            if raw_ndvi["NDVI"].max() > 1.5:
                raw_ndvi["NDVI"] *= 0.0001
            raw_ndvi = raw_ndvi.dropna(subset=["Date", "NDVI"])
            raw_ndvi = raw_ndvi[(raw_ndvi["NDVI"] >= -0.2) & (raw_ndvi["NDVI"] <= 1.0)].copy()
            raw_ndvi["Year"] = raw_ndvi["Date"].dt.year
            raw_ndvi["Month"] = raw_ndvi["Date"].dt.month
            ndvi_df = raw_ndvi[raw_ndvi["Year"] <= 2025].copy()

    return merged[merged["Year"] <= 2025].copy(), ndvi_df


df_complete, df_ndvi = load_data()

st.title("🌾 The story of Shakargarh's resources: Crop-Water stress in the past 4 decades")
st.markdown(
    "**NASA 2026 Space Apps Challenge: *Be An Earth System Trend Detective!***\n\n"
    "*For generations, farming families between the River Ravi and Nullah Bein have timed their sowing and harvesting to the rhythm of the seasons. "
    "Combining 44 years of NASA FLDAS hydrological records (`1982–2025`) with 26 years of 1 km Terra MODIS satellite crop greenness (`2000–2025`), "
    "we trace how rising atmospheric demand and changing soil moisture affect crop production in Shakargarh.*"
)

st.sidebar.header("🧭 Explore Shakargarh's Seasons")
season = st.sidebar.selectbox("1. Choose a Farming Season", ["Full Agricultural Year (Annual Mean)", "Kharif Season (Jun–Sep: Monsoon & Rice)", "Rabi Season (Nov–Apr: Winter Wheat)"])
split_year = st.sidebar.slider("2. Compare Before vs. After Year", 1995, 2015, 2004)
st.sidebar.markdown("---")
st.sidebar.caption("**Why this matters:** Faster evaporation under warmer skies can leave crops thirstier during dry months—a pattern known as progressive aridification.")

if season == "Full Agricultural Year (Annual Mean)":
    months_filter = list(range(1, 13))
    crop_note = "Over the past four decades, atmospheric water pull has risen 18.52% (p ≈ 0.01), slightly outpacing root-zone soil moisture gains (+17.95%, p ≈ 0.04)."
elif season == "Kharif Season (Jun–Sep: Monsoon & Rice)":
    months_filter = [6, 7, 8, 9]
    crop_note = "During the summer monsoon, root-zone soil moisture rises significantly (p < 0.05), yet crop evapotranspiration levels off—showing excess water often waterlogs paddy fields."
else:
    months_filter = [11, 12, 1, 2, 3, 4]
    crop_note = "Winter wheat shows no statistically significant soil moisture gain (p > 0.05) over 44 years, leaving Rabi crops sensitive to dry-season evaporative stress."

sub_m = df_complete[df_complete["Month"].isin(months_filter)].copy()
ann = sub_m.groupby("Year", as_index=False).mean(numeric_only=True)

mk_et = safe_mk(ann["ET_mm_month"])
mk_sm = safe_mk(ann["SM_10_40cm"])
mk_ar = safe_mk(ann["Aridity_Stress_Idx"])
r_val, p_corr = pearsonr(sub_m["SM_10_40cm"], sub_m["ET_mm_month"])

early, late = ann[ann["Year"] < split_year], ann[ann["Year"] >= split_year]
et_shift = ((late["ET_mm_month"].mean() - early["ET_mm_month"].mean()) / early["ET_mm_month"].mean()) * 100.0
sm_shift = ((late["SM_10_40cm"].mean() - early["SM_10_40cm"].mean()) / early["SM_10_40cm"].mean()) * 100.0
ar_shift = ((late["Aridity_Stress_Idx"].mean() - early["Aridity_Stress_Idx"].mean()) / early["Aridity_Stress_Idx"].mean()) * 100.0

st.info(f"**What the data tells us ({season}):** {crop_note}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Crop Water Demand (ET)", f"{et_shift:+.2f}%", f"p = {mk_et.p:.4f} ({'Significant' if mk_et.p < 0.05 else 'Variable'})")
c2.metric("Root-Zone Soil Water (10–40 cm)", f"{sm_shift:+.2f}%", f"p = {mk_sm.p:.4f} ({'Significant' if mk_sm.p < 0.05 else 'Variable'})")
c3.metric("Crop-Water Stress Ratio", f"{ar_shift:+.2f}%", f"Slope: {mk_ar.slope:+.5f}/yr")
c4.metric("Soil-to-Crop Link (r)", f"{r_val:.3f}", f"p = {p_corr:.2e}")

tab1, tab2, tab3 = st.tabs(["📈 4-Decade Water Stress Trends", "🛰️ 1 km Satellite Crop Greenness (MODIS NDVI)", "🌱 What This Means for Local Farmers"])

with tab1:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            f"1. Crop & Soil Water Loss (Evapotranspiration, mm/mo) | Change: {mk_et.slope:+.4f}/yr (p = {mk_et.p:.4f})",
            f"2. Root-Zone Soil Moisture (10–40 cm depth, m³/m³) | Change: {mk_sm.slope:+.5f}/yr (p = {mk_sm.p:.4f})",
            f"3. Crop-Water Stress Index (Water Demand vs. Soil Supply) | Change: {mk_ar.slope:+.5f}/yr (p = {mk_ar.p:.4f})",
        ),
    )
    for row_i, (col_n, col_hex, dash_hex) in enumerate([("ET_mm_month", "#c84b31", "#6b1d0e"), ("SM_10_40cm", "#005a9c", "#002b49"), ("Aridity_Stress_Idx", "#d97706", "#78350f")], start=1):
        fig.add_trace(go.Scatter(x=ann["Year"], y=ann[col_n], mode="lines+markers", line=dict(color=col_hex, width=2)), row=row_i, col=1)
        z = np.polyfit(ann["Year"], ann[col_n], 1)
        fig.add_trace(go.Scatter(x=ann["Year"], y=np.poly1d(z)(ann["Year"]), mode="lines", line=dict(color=dash_hex, dash="dash")), row=row_i, col=1)

    fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
    fig.update_layout(height=740, template="plotly_white", showlegend=False, margin=dict(t=45, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Evidence (Autocorrelation-Corrected Mann-Kendall Test)")
    st.table(pd.DataFrame({
        "Resource Indicator": ["Evapotranspiration (Crop & Soil Water Use)", "Root-Zone Soil Moisture (10–40 cm)", "Crop-Water Stress Index"],
        "44-Year Direction": [mk_et.trend.capitalize(), mk_sm.trend.capitalize(), mk_ar.trend.capitalize()],
        "Annual Rate of Change": [f"{mk_et.slope:+.4f} mm/mo", f"{mk_sm.slope:+.5f} m³/m³", f"{mk_ar.slope:+.5f} /yr"],
        "p-value": [f"{mk_et.p:.4f}", f"{mk_sm.p:.4f}", f"{mk_ar.p:.4f}"],
        "Statistically Proven (p < 0.05)": ["Yes" if mk_et.p < 0.05 else "No (High Swing)", "Yes" if mk_sm.p < 0.05 else "No (High Swing)", "Yes" if mk_ar.p < 0.05 else "No (High Swing)"],
        f"Change After {split_year}": [f"{et_shift:+.2f}%", f"{sm_shift:+.2f}%", f"{ar_shift:+.2f}%"],
    }))

with tab2:
    if df_ndvi is None or df_ndvi.empty:
        st.info("Upload your NASA AppEEARS `MOD13A3` CSV file to activate 1 km satellite crop greenness trends.")
    else:
        st.subheader("26-Year Satellite Crop Greenness Across Shakargarh's Agricultural Zones (2000–2025)")
        ndvi_sub = df_ndvi[df_ndvi["Month"].isin(months_filter)].copy()
        id_col = "ID" if "ID" in ndvi_sub.columns else ("Category" if "Category" in ndvi_sub.columns else None)
        fig_ndvi = go.Figure()
        colors = ["#1b9e77", "#7570b3", "#e7298a"]
        zone_stats = []
        if id_col:
            for idx, (zone_name, grp) in enumerate(ndvi_sub.groupby(id_col)):
                ann_z = grp.groupby("Year", as_index=False)["NDVI"].mean()
                mk_z = safe_mk(ann_z["NDVI"])
                fig_ndvi.add_trace(go.Scatter(x=ann_z["Year"], y=ann_z["NDVI"], mode="lines+markers", name=f"{zone_name} (p={mk_z.p:.3f})", line=dict(color=colors[idx % 3], width=2.2)))
                zone_stats.append({
                    "Shakargarh Agricultural Zone": zone_name,
                    "Mean NDVI": f"{ann_z['NDVI'].mean():.3f}",
                    "26-Yr Trend": mk_z.trend.capitalize(),
                    "Annual Slope": f"{mk_z.slope:+.5f}/yr",
                    "p-value": f"{mk_z.p:.4f}",
