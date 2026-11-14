from pathlib import Path

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

DATA_DIR = Path(__file__).resolve().parent
MAX_YEAR = 2025
SEASON_OPTIONS = [
    "Full Agricultural Year (Annual Mean)",
    "Kharif Season (Jun–Sep: Monsoon & Rice)",
    "Rabi Season (Nov–Apr: Winter Wheat)",
]


def read_giovanni(pattern: str, column_name: str) -> pd.DataFrame:
    """Load a Giovanni CSV with a metadata preamble and time-series table."""
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
    """Load MODIS NDVI if an appropriately formatted CSV is available."""
    matches = sorted(DATA_DIR.glob("*MOD13A3*.csv"))
    if not matches:
        return None

    raw = pd.read_csv(matches[0], encoding="utf-8-sig")
    date_column = next((c for c in raw.columns if c.strip().lower() == "date"), None)
    ndvi_column = next(
        (c for c in raw.columns if "NDVI" in c and "Quality" not in c and "VI_Quality" not in c),
        None,
    )
    if date_column is None or ndvi_column is None:
        return None

    result = raw.copy()
    result["Date"] = pd.to_datetime(result[date_column], errors="coerce")
    result["NDVI"] = pd.to_numeric(result[ndvi_column], errors="coerce")
    if result["NDVI"].max(skipna=True) > 1.5:
        result["NDVI"] *= 0.0001
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
        raise ValueError("The ET and soil-moisture files have no dates in common.")

    # Giovanni ET is a monthly mean flux; convert to depth using the actual
    # number of seconds in each calendar month.
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
    pairs = frame[[x_column, y_column]].dropna()
    if len(pairs) < 2 or pairs[x_column].nunique() < 2 or pairs[y_column].nunique() < 2:
        return np.nan, np.nan
    return pearsonr(pairs[x_column], pairs[y_column])


def percent_change(before, after):
    if pd.isna(before) or pd.isna(after) or np.isclose(before, 0):
        return np.nan
    return (after - before) / before * 100.0


def format_change(value):
    return f"{value:+.2f}%" if pd.notna(value) else "N/A"


def test_description(result, indicator):
    if result is None:
        return f"There is insufficient variation or yearly data to estimate a reliable {indicator} trend."
    direction = result.trend.lower()
    evidence = "statistically significant at the 5% level" if result.p < 0.05 else "not statistically significant at the 5% level"
    return (
        f"The annual/seasonal series indicates a {direction} in {indicator} "
        f"(Sen slope {result.slope:+.5g} per year; p = {result.p:.4f}), {evidence}."
    )


# Restore the original project title.
st.title("🌾 The story of Shakargarh's resources: Crop-Water stress in the past 4 decades")
st.markdown(
    """### Research overview
    This dashboard examines long-term seasonal variability in modeled evapotranspiration (ET) and
    root-zone soil moisture over Shakargarh using NASA FLDAS monthly area-average time series. Where
    available, Terra MODIS NDVI is included as a satellite-derived vegetation greenness indicator.
    Comparing ET with soil moisture describes how atmospheric water use and modeled root-zone supply
    co-vary; NDVI provides complementary context on vegetation condition. The analysis summarizes
    annual and crop-season patterns through 2025 and reports autocorrelation-adjusted
    Mann–Kendall trend evidence. These associations are descriptive and do not establish causation,
    crop yield impacts, or field-level water availability."""
)

with st.sidebar:
    st.header("🧭 Explore Shakargarh's seasons")
    season = st.selectbox("Choose a farming season", SEASON_OPTIONS)
    split_year = st.slider("Compare periods at year", 1995, 2015, 2004)
    st.caption("Rabi spans November–April and is grouped by the year in which the season ends.")

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
    season_note = "Annual averages use all available months in each calendar year."
elif season == SEASON_OPTIONS[1]:
    months_filter = [6, 7, 8, 9]
    selected = df_complete[df_complete["Month"].isin(months_filter)].copy()
    selected["SeasonYear"] = selected["Year"]
    season_note = "Kharif summarizes June–September observations by calendar year."
else:
    months_filter = [11, 12, 1, 2, 3, 4]
    selected = df_complete[df_complete["Month"].isin(months_filter)].copy()
    selected["SeasonYear"] = selected["Year"] + selected["Month"].isin([11, 12]).astype(int)
    season_note = "Rabi summarizes November–April, grouped by the year in which the season ends."

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
mk_stress = safe_mann_kendall(annual["Aridity_Stress_Idx"])
r_value, p_value = safe_correlation(selected, "SM_10_40cm", "ET_mm_month")

st.caption(f"**Selected period:** {season_note}")
if before.empty or after.empty:
    st.warning("One comparison period is empty for this year selection; percentage changes are shown as N/A.")

st.subheader("Summary statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ET change after split year", format_change(et_change))
c2.metric("Root-zone moisture change", format_change(sm_change))
c3.metric("Relative demand index change", format_change(stress_change))
c4.metric(
    "Monthly moisture–ET correlation",
    f"r = {r_value:.3f}" if pd.notna(r_value) else "N/A",
    f"Pearson p = {p_value:.2e}" if pd.notna(p_value) else None,
)

with st.expander("Methods and interpretation"):
    st.markdown(
        """- **ET:** Giovanni monthly mean flux converted to monthly depth using each month's actual duration.
        - **Root-zone moisture:** FLDAS soil moisture for the 10–40 cm layer.
        - **Relative demand index:** ET divided by root-zone moisture scaled by 300. It is an exploratory ratio, not a validated drought or crop-stress index.
        - **Trend test:** Hamed–Rao modified Mann–Kendall test applied to yearly/seasonal averages. A p-value below 0.05 indicates statistical evidence of a monotonic trend under the test assumptions; it does not identify a cause.
        - **Spatial and practical scope:** FLDAS values are modeled area averages. They should guide questions and monitoring priorities, not replace field measurements, irrigation records, rainfall data, or agronomic advice."""
    )

st.subheader("📈 Seasonal water trends and local interpretation")
chart_column, interpretation_column = st.columns([1.75, 1], gap="large")
with chart_column:
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
                go.Scatter(
                    x=points["Year"], y=points[column], mode="lines+markers",
                    line={"color": line_color}, name=column, showlegend=False,
                ),
                row=row,
                col=1,
            )
            if len(points) >= 2:
                slope, intercept = np.polyfit(points["Year"], points[column], 1)
                fig.add_trace(
                    go.Scatter(
                        x=points["Year"], y=slope * points["Year"] + intercept,
                        mode="lines", line={"color": trend_color, "dash": "dash"},
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
        fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
        fig.update_layout(
            height=700, template="plotly_white", showlegend=False,
            hovermode="x unified", margin={"t": 55, "b": 35, "l": 25, "r": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

with interpretation_column:
    st.markdown("#### What the selected data indicates")
    st.write(
        f"**ET:** {format_change(et_change)} comparing the mean before {split_year} with the mean from {split_year} onward. "
        + test_description(mk_et, "evapotranspiration")
    )
    st.write(
        f"**Root-zone moisture:** {format_change(sm_change)} across the same comparison. "
        + test_description(mk_sm, "root-zone soil moisture")
    )
    st.write(
        f"**Relative demand:** {format_change(stress_change)}. "
        + test_description(mk_stress, "the exploratory relative-demand index")
    )
    st.write(
        f"For the selected months, monthly soil moisture and ET have Pearson r = "
        f"{r_value:.3f} (p = {p_value:.2e})" if pd.notna(r_value)
        else "There are insufficient paired observations or variation to estimate the monthly soil-moisture–ET correlation."
    )

    st.markdown("#### Practical use for farmers and researchers")
    st.markdown(
        """- **For farmers:** Use the seasonal pattern to decide when to check field soil moisture and crop condition more frequently. Compare the regional signal with field observations before changing irrigation timing or amounts.
        - **For researchers and extension teams:** Treat divergence between ET and root-zone moisture as a monitoring question. Validate it against rainfall, irrigation, crop calendars, and local soil measurements.
        - **Interpret cautiously:** A rise in modeled soil moisture does not by itself establish more usable water, and ET is not a direct measurement of crop water requirement or yield."""
    )

st.subheader("Statistical evidence")
trend_rows = []
for name, result in [
    ("Evapotranspiration", mk_et),
    ("Root-zone soil moisture", mk_sm),
    ("Relative demand index", mk_stress),
]:
    trend_rows.append(
        {
            "Indicator": name,
            "Direction": result.trend.capitalize() if result else "Insufficient data/variation",
            "Sen slope per year": f"{result.slope:+.5g}" if result else "N/A",
            "p-value": f"{result.p:.4f}" if result else "N/A",
            "Significant at p < 0.05": "Yes" if result and result.p < 0.05 else "No" if result else "N/A",
        }
    )
st.dataframe(pd.DataFrame(trend_rows), hide_index=True, use_container_width=True)

st.subheader("🛰️ MODIS satellite vegetation greenness (NDVI)")
if df_ndvi is None or df_ndvi.empty:
    st.info("No readable MOD13A3 NDVI CSV was found. The FLDAS water analysis remains available.")
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
                    "Trend": result.trend.capitalize() if result else "Insufficient data/variation",
                    "Sen slope per year": f"{result.slope:+.5f}" if result else "N/A",
                    "p-value": f"{result.p:.4f}" if result else "N/A",
                }
            )
        ndvi_chart.update_layout(
            height=430, template="plotly_white", yaxis_title="NDVI",
            xaxis_title="Year", hovermode="x unified",
        )
        st.plotly_chart(ndvi_chart, use_container_width=True)
        st.dataframe(pd.DataFrame(ndvi_rows), hide_index=True, use_container_width=True)
        st.caption(
            "NDVI is a satellite-derived vegetation greenness measure. It is not a direct measure of crop yield, "
            "and cross-sensor or crop-stage effects should be considered when interpreting changes."
        )

st.caption(
    f"Sources: NASA Giovanni FLDAS monthly area-average time series and Terra MODIS MOD13A3 NDVI where available. "
    f"Analysis includes observations through {MAX_YEAR}."
)
