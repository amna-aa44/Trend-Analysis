from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

st.set_page_config(
    page_title="Shakargarh Crop–Water Trends",
    page_icon="🌾",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent
MAX_YEAR = 2025
SEASON_OPTIONS = [
    "Full Agricultural Year (Annual Mean)",
    "Kharif Season (Jun–Sep: Monsoon & Rice)",
    "Rabi Season (Nov–Apr: Winter Wheat)",
]


def read_giovanni(pattern: str, column_name: str) -> pd.DataFrame:
    """Load a Giovanni CSV that has a metadata preamble and a time-series table."""
    matches = sorted(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing CSV matching {pattern!r} in {DATA_DIR}")

    path = matches[0]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        lines = source.readlines()

    header_row = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("time,")),
        None,
    )
    if header_row is None:
        raise ValueError(f"Could not find the Giovanni time-series header in {path.name}")

    frame = pd.read_csv(path, skiprows=header_row)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected date and value columns in {path.name}")

    frame = frame.iloc[:, :2].copy()
    frame.columns = ["Date", column_name]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame[column_name] = frame[column_name].replace([-9999, -9999.0], np.nan)
    return frame.dropna(subset=["Date", column_name]).drop_duplicates("Date")


def read_ndvi() -> pd.DataFrame | None:
    """Load optional MODIS NDVI CSV if the repository contains one."""
    matches = sorted(DATA_DIR.glob("*MOD13A3*.csv"))
    if not matches:
        return None

    raw = pd.read_csv(matches[0], encoding="utf-8-sig")
    date_column = next((c for c in raw.columns if c.strip().lower() == "date"), None)
    ndvi_column = next(
        (
            c for c in raw.columns
            if "NDVI" in c and "Quality" not in c and "VI_Quality" not in c
        ),
        None,
    )
    if date_column is None or ndvi_column is None:
        return None

    result = raw.copy()
    result["Date"] = pd.to_datetime(result[date_column], errors="coerce")
    result["NDVI"] = pd.to_numeric(result[ndvi_column], errors="coerce")
    # MOD13A3 NDVI is often stored as a scaled integer; leave already-scaled
    # values unchanged and convert values in the documented integer range.
    if result["NDVI"].max(skipna=True) > 1.5:
        result["NDVI"] = result["NDVI"] * 0.0001
    result = result.dropna(subset=["Date", "NDVI"])
    result = result[result["NDVI"].between(-0.2, 1.0)].copy()
    result["Year"] = result["Date"].dt.year
    result["Month"] = result["Date"].dt.month
    if "ID" not in result.columns:
        result["ID"] = result["Category"] if "Category" in result.columns else "Shakargarh"
    return result[result["Year"] <= MAX_YEAR].copy()


@st.cache_data(show_spinner="Loading NASA Giovanni datasets…")
def load_data():
    et = read_giovanni("*Evap_tavg*.csv", "ET_raw")
    surface = read_giovanni("*SoilMoi00_10cm*.csv", "SM_0_10cm")
    root = read_giovanni("*SoilMoi10_40cm*.csv", "SM_10_40cm")

    water = et.merge(surface, on="Date", how="inner").merge(root, on="Date", how="inner")
    if water.empty:
        raise ValueError("The ET and soil-moisture CSV files have no dates in common.")

    # Convert the Giovanni monthly mean flux to monthly depth using each
    # calendar month's actual duration.
    seconds_per_month = water["Date"].dt.days_in_month * 24 * 60 * 60
    water["ET_mm_month"] = water["ET_raw"] * seconds_per_month
    water["Aridity_Stress_Idx"] = np.where(
        water["SM_10_40cm"] > 0,
        water["ET_mm_month"] / (water["SM_10_40cm"] * 300.0),
        np.nan,
    )
    water["Year"] = water["Date"].dt.year
    water["Month"] = water["Date"].dt.month
    water = water[water["Year"] <= MAX_YEAR].copy()
    return water, read_ndvi()


def safe_mann_kendall(series):
    values = pd.Series(series).dropna()
    if len(values) < 3 or values.nunique() < 2:
        return None
    try:
        return mk.hamed_rao_modification_test(values.to_numpy())
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None


def safe_correlation(frame, x_column, y_column):
    pair = frame[[x_column, y_column]].dropna()
    if len(pair) < 2 or pair[x_column].nunique() < 2 or pair[y_column].nunique() < 2:
        return np.nan, np.nan
    return pearsonr(pair[x_column], pair[y_column])


def percent_change(before, after):
    if pd.isna(before) or pd.isna(after) or np.isclose(before, 0):
        return np.nan
    return (after - before) / before * 100.0


def format_change(value):
    return f"{value:+.2f}%" if pd.notna(value) else "N/A"


st.title("🌾 Shakargarh’s crop–water story")
st.markdown(
    """**Seasonal water demand, root-zone moisture, and satellite crop greenness**

    Explore NASA FLDAS monthly estimates for Shakargarh, with optional Terra MODIS NDVI observations. "
    "These are modeled and area-averaged indicators—not direct measurements of crop yield or "
    "conditions on an individual farm."""
)

with st.sidebar:
    st.header("🧭 Explore Shakargarh’s seasons")
    season = st.selectbox("Choose a farming season", SEASON_OPTIONS)
    split_year = st.slider("Compare periods at year", 1995, 2015, 2004)
    st.caption("Rabi runs November–April and is grouped by the year in which the season ends.")

try:
    df_complete, df_ndvi = load_data()
except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as error:
    st.error(f"The app could not load its data: {error}")
    st.info("Keep the three FLDAS Giovanni CSV files in the same repository folder as app.py.")
    st.stop()

if df_complete.empty:
    st.error(f"No valid FLDAS observations are available through {MAX_YEAR}.")
    st.stop()

if season == SEASON_OPTIONS[0]:
    months_filter = list(range(1, 13))
    selected = df_complete.copy()
    selected["SeasonYear"] = selected["Year"]
    note = "Annual averages summarize all available months in each calendar year."
elif season == SEASON_OPTIONS[1]:
    months_filter = [6, 7, 8, 9]
    selected = df_complete[df_complete["Month"].isin(months_filter)].copy()
    selected["SeasonYear"] = selected["Year"]
    note = "Kharif summarizes June–September observations by calendar year."
else:
    months_filter = [11, 12, 1, 2, 3, 4]
    selected = df_complete[df_complete["Month"].isin(months_filter)].copy()
    # A Rabi season crosses the calendar-year boundary; Nov/Dec belong to the
    # season ending in the following year.
    selected["SeasonYear"] = selected["Year"] + selected["Month"].isin([11, 12]).astype(int)
    note = "Rabi summarizes November–April, grouped by the year in which the season ends."

if selected.empty:
    st.warning("No observations are available for the selected season.")
    st.stop()

annual = (
    selected.groupby("SeasonYear")[["ET_mm_month", "SM_10_40cm", "Aridity_Stress_Idx"]]
    .mean()
    .reset_index()
    .rename(columns={"SeasonYear": "Year"})
)
if annual.empty:
    st.warning("No annual or seasonal summary data could be calculated.")
    st.stop()

before = annual[annual["Year"] < split_year]
after = annual[annual["Year"] >= split_year]

def group_mean(frame, column):
    return frame[column].mean() if not frame.empty else np.nan

et_change = percent_change(group_mean(before, "ET_mm_month"), group_mean(after, "ET_mm_month"))
sm_change = percent_change(group_mean(before, "SM_10_40cm"), group_mean(after, "SM_10_40cm"))
stress_change = percent_change(
    group_mean(before, "Aridity_Stress_Idx"), group_mean(after, "Aridity_Stress_Idx")
)

mk_et = safe_mann_kendall(annual["ET_mm_month"])
mk_sm = safe_mann_kendall(annual["SM_10_40cm"])
mk_ar = safe_mann_kendall(annual["Aridity_Stress_Idx"])
r_value, p_value = safe_correlation(selected, "SM_10_40cm", "ET_mm_month")

st.caption(note)
if before.empty or after.empty:
    st.warning("One comparison period is empty for this year selection; percentage changes appear as N/A.")
st.subheader("At a glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Evapotranspiration shift", format_change(et_change))
c2.metric("Root-zone soil moisture shift", format_change(sm_change))
c3.metric("Relative demand index shift", format_change(stress_change))
c4.metric(
    "Soil moisture ↔ ET correlation",
    f"r = {r_value:.3f}" if pd.notna(r_value) else "N/A",
    f"Pearson p = {p_value:.2e}" if pd.notna(p_value) else None,
)
st.info(f"**Selected view ({season}):** {note}")

with st.expander("Indicator notes and limitations"):
    st.markdown(
        """- **ET:** Monthly evapotranspiration depth derived from FLDAS mean flux and actual month length.
        - **Root-zone moisture:** FLDAS soil moisture for the 10–40 cm layer.
        - **Relative demand index:** An exploratory ET-to-moisture ratio, not a validated crop-stress or drought index.
        - **Trend test:** Hamed–Rao modified Mann–Kendall on annual/seasonal means. Significance does not establish cause or predict future conditions."""
    )

with st.container(border=True):
    st.subheader("📈 Four-decade water trends")
    if len(annual) < 2:
        st.info("At least two years of data are needed to draw trend lines.")
    else:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Evapotranspiration (mm/month)",
                "Root-zone soil moisture (10–40 cm)",
                "Relative demand index (exploratory)",
            ),
        )
        chart_series = [
            ("ET_mm_month", "#c84b31", "#6b1d0e"),
            ("SM_10_40cm", "#005a9c", "#002b49"),
            ("Aridity_Stress_Idx", "#d97706", "#78350f"),
        ]
        for row, (column, line_color, trend_color) in enumerate(chart_series, start=1):
            points = annual[["Year", column]].dropna()
            fig.add_trace(
                go.Scatter(x=points["Year"], y=points[column], mode="lines+markers", line={"color": line_color}),
                row=row,
                col=1,
            )
            if len(points) >= 2:
                slope, intercept = np.polyfit(points["Year"], points[column], 1)
                fig.add_trace(
                    go.Scatter(
                        x=points["Year"],
                        y=slope * points["Year"] + intercept,
                        mode="lines",
                        line={"color": trend_color, "dash": "dash"},
                    ),
                    row=row,
                    col=1,
                )
        fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
        fig.update_layout(height=740, template="plotly_white", showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical evidence")
    trend_rows = []
    for name, result in [
        ("Evapotranspiration", mk_et),
        ("Root-zone soil moisture", mk_sm),
        ("Relative demand index", mk_ar),
    ]:
        trend_rows.append(
            {
                "Indicator": name,
                "Direction": result.trend.capitalize() if result else "Not enough variation/data",
                "Sen slope": f"{result.slope:+.5g}" if result else "N/A",
                "p-value": f"{result.p:.4f}" if result else "N/A",
                "Significant (p < 0.05)": "Yes" if result and result.p < 0.05 else "No" if result else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(trend_rows), hide_index=True, use_container_width=True)

with st.container(border=True):
    st.subheader("🛰️ MODIS satellite crop greenness (NDVI)")
    if df_ndvi is None or df_ndvi.empty:
        st.info("No readable MOD13A3 NDVI CSV was found. Water-trend analysis works without it.")
    else:
        ndvi_selected = df_ndvi[df_ndvi["Month"].isin(months_filter)].copy()
        if ndvi_selected.empty:
            st.info("The NDVI file has no records for the selected months.")
        else:
            ndvi_chart = go.Figure()
            ndvi_rows = []
            for zone, group in ndvi_selected.groupby("ID"):
                yearly = group.groupby("Year", as_index=False)["NDVI"].mean()
                result = safe_mann_kendall(yearly["NDVI"])
                ndvi_chart.add_trace(
                    go.Scatter(x=yearly["Year"], y=yearly["NDVI"], mode="lines+markers", name=str(zone))
                )
                ndvi_rows.append(
                    {
                        "Zone": zone,
                        "Mean NDVI": f"{yearly['NDVI'].mean():.3f}",
                        "Trend": result.trend.capitalize() if result else "Not enough variation/data",
                        "Slope/year": f"{result.slope:+.5f}" if result else "N/A",
                        "p-value": f"{result.p:.4f}" if result else "N/A",
                    }
                )
            ndvi_chart.update_layout(
                height=430,
                template="plotly_white",
                yaxis_title="NDVI",
                xaxis_title="Year",
                hovermode="x unified",
            )
            st.plotly_chart(ndvi_chart, use_container_width=True)
            st.dataframe(pd.DataFrame(ndvi_rows), hide_index=True, use_container_width=True)

            ndvi_monthly = ndvi_selected.groupby(["Year", "Month"], as_index=False)["NDVI"].mean()
            water_ndvi = selected.merge(ndvi_monthly, on=["Year", "Month"], how="inner")
            r_ndvi, p_ndvi = safe_correlation(water_ndvi, "SM_10_40cm", "NDVI")
            if pd.notna(r_ndvi):
                st.write(f"Monthly root-zone moisture vs. NDVI: r = {r_ndvi:.3f}, p = {p_ndvi:.2e}")

st.subheader("🌱 What this means for local farming")
st.markdown(
    """Season-by-season comparisons can help identify when modeled water demand and root-zone moisture move together or diverge. Kharif monsoon months and the drier Rabi growing season have different water-management contexts.

    Use these regional results as context, not field-level advice: the dashboard does not include rainfall, irrigation, crop yields, or on-farm measurements. Local observations and farmer experience are essential for interpreting trends."""
)
st.caption(
    f"Source: NASA Giovanni FLDAS monthly area-average time series. Analysis through {MAX_YEAR}; "
    "MODIS NDVI is displayed when a compatible file is present."
)
