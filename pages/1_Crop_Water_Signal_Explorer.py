from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

st.set_page_config(
    page_title="Seasonal Crop-Water Signal Explorer",
    page_icon="🌱",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parents[1]
MAX_YEAR = 2025
SEASONS = {
    "Full year": list(range(1, 13)),
    "Kharif (Jun–Sep)": [6, 7, 8, 9],
    "Rabi (Nov–Apr)": [11, 12, 1, 2, 3, 4],
}


def read_giovanni(pattern: str, column_name: str) -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing FLDAS CSV matching {pattern!r} in {DATA_DIR}")
    path = matches[0]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        lines = source.readlines()
    header = next(
        (index for index, line in enumerate(lines) if line.strip().lower().startswith("time,")),
        None,
    )
    if header is None:
        raise ValueError(f"Could not find the Giovanni time-series header in {path.name}")
    frame = pd.read_csv(path, skiprows=header)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected date and value columns in {path.name}")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["Date", column_name]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame[column_name] = frame[column_name].replace([-9999, -9999.0], np.nan)
    return frame.dropna(subset=["Date", column_name]).drop_duplicates("Date")


@st.cache_data(show_spinner="Loading FLDAS and MODIS time series…")
def load_dashboard_data():
    et = read_giovanni("*Evap_tavg*.csv", "ET_raw")
    root = read_giovanni("*SoilMoi10_40cm*.csv", "SM_10_40cm")
    water = et.merge(root, on="Date", how="inner")
    if water.empty:
        raise ValueError("The FLDAS ET and root-zone soil-moisture files have no dates in common.")
    seconds_per_month = water["Date"].dt.days_in_month * 24 * 60 * 60
    water["ET_mm_month"] = water["ET_raw"] * seconds_per_month
    water["Year"] = water["Date"].dt.year
    water["Month"] = water["Date"].dt.month
    water = water[water["Year"] <= MAX_YEAR].copy()

    files = sorted(DATA_DIR.glob("*MOD13A3*.csv"))
    if not files:
        raise FileNotFoundError("No MOD13A3 CSV was found beside the app.")
    raw = pd.read_csv(files[0], encoding="utf-8-sig")
    date_column = next((column for column in raw.columns if column.strip().lower() == "date"), None)
    ndvi_column = next(
        (
            column for column in raw.columns
            if "ndvi" in column.lower() and "quality" not in column.lower()
        ),
        None,
    )
    if date_column is None or ndvi_column is None:
        raise ValueError("The MOD13A3 CSV needs a Date column and an NDVI data column.")

    ndvi = pd.DataFrame()
    ndvi["Date"] = pd.to_datetime(raw[date_column], errors="coerce")
    ndvi["NDVI"] = pd.to_numeric(raw[ndvi_column], errors="coerce")
    if ndvi["NDVI"].max(skipna=True) > 1.5:
        ndvi["NDVI"] *= 0.0001
    id_column = next((column for column in raw.columns if column.strip().lower() == "id"), None)
    category_column = next((column for column in raw.columns if column.strip().lower() == "category"), None)
    if id_column:
        ndvi["Zone"] = raw[id_column].astype("string").str.strip()
    elif category_column:
        ndvi["Zone"] = raw[category_column].astype("string").str.strip()
    else:
        ndvi["Zone"] = "Shakargarh"
    ndvi = ndvi.dropna(subset=["Date", "NDVI", "Zone"])
    ndvi = ndvi[ndvi["NDVI"].between(-0.2, 1.0) & (ndvi["Date"].dt.year <= MAX_YEAR)].copy()
    ndvi["Year"] = ndvi["Date"].dt.year
    ndvi["Month"] = ndvi["Date"].dt.month
    return water, ndvi


def add_season_year(frame: pd.DataFrame, season_name: str) -> pd.DataFrame:
    result = frame.copy()
    if season_name == "Rabi (Nov–Apr)":
        result["SeasonYear"] = result["Year"] + result["Month"].isin([11, 12]).astype(int)
    else:
        result["SeasonYear"] = result["Year"]
    return result


def correlation(x: pd.Series, y: pd.Series):
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(pairs["x"], pairs["y"])
    return result.statistic, result.pvalue


st.title("🌱 Seasonal crop-water and vegetation signal explorer")
st.markdown(
    "Compare NASA FLDAS modeled evapotranspiration (ET) and root-zone soil moisture with "
    "MODIS MOD13A3 1-km NDVI for the three representative Shakargarh cropland zones in the input file. "
    "This is a **screening dashboard for co-occurring signals**, not a crop-stress diagnosis or a field-level alert."
)

try:
    water, ndvi = load_dashboard_data()
except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the dashboard data: {error}")
    st.info("Keep the FLDAS Giovanni CSV files and MOD13A3 results CSV in the repository root.")
    st.stop()

zones = sorted(ndvi["Zone"].dropna().unique().tolist())
if len(zones) != 3:
    st.warning(
        f"The loaded NDVI file contains {len(zones)} distinct zone/group names, not three. "
        "The dashboard will show the groups actually present; verify that the input CSV contains "
        "the intended three representative cropland zones."
    )

with st.sidebar:
    st.header("Explore crop-water signals")
    season = st.selectbox("Season", list(SEASONS))
    available_years = sorted(set(water["Year"]).intersection(set(ndvi["Year"])))
    if available_years:
        year_range = st.slider(
            "Year range",
            min_value=int(min(available_years)),
            max_value=int(max(available_years)),
            value=(int(min(available_years)), int(max(available_years))),
        )
    else:
        year_range = None
    st.caption("Rabi seasons are assigned to the year in which they end.")

months = SEASONS[season]
water_selected = water[water["Month"].isin(months)].copy()
ndvi_selected = ndvi[ndvi["Month"].isin(months)].copy()
if year_range:
    water_selected = water_selected[water_selected["Year"].between(*year_range)]
    ndvi_selected = ndvi_selected[ndvi_selected["Year"].between(*year_range)]

if water_selected.empty or ndvi_selected.empty:
    st.warning(
        f"There are no paired FLDAS and NDVI observations for **{season}** in the selected year range. "
        "The dashboard cannot make a seasonal comparison from the available records."
    )
    observed_months = sorted(ndvi["Month"].unique().tolist())
    month_labels = [pd.Timestamp(2000, month, 1).strftime("%b") for month in observed_months]
    st.info("NDVI months present in this file: " + (", ".join(month_labels) if month_labels else "none"))
    st.stop()

water_selected = add_season_year(water_selected, season)
ndvi_selected = add_season_year(ndvi_selected, season)
water_yearly = (
    water_selected.groupby("SeasonYear", as_index=False)
    .agg(ET_mm_month=("ET_mm_month", "mean"), SM_10_40cm=("SM_10_40cm", "mean"),
         FLDAS_months=("Month", "nunique"))
)
ndvi_yearly = (
    ndvi_selected.groupby(["Zone", "SeasonYear"], as_index=False)
    .agg(NDVI=("NDVI", "mean"), NDVI_months=("Month", "nunique"))
)
paired = ndvi_yearly.merge(water_yearly, on="SeasonYear", how="inner")
if paired.empty:
    st.warning("No season-year pairs could be aligned between the FLDAS and MODIS records.")
    st.stop()

observed_months = sorted(ndvi_selected["Month"].unique().tolist())
month_labels = [pd.Timestamp(2000, month, 1).strftime("%b") for month in observed_months]
coverage_col, zones_col, years_col = st.columns(3)
coverage_col.metric("NDVI months represented", ", ".join(month_labels))
zones_col.metric("MODIS groups", str(ndvi_selected["Zone"].nunique()))
years_col.metric("Paired season-years", str(paired["SeasonYear"].nunique()))
if len(observed_months) < len(months):
    st.warning(
        f"NDVI covers only {', '.join(month_labels)} for the selected period, rather than every month "
        f"in {season}. Seasonal vegetation patterns are therefore only partially represented; "
        "interpret comparisons using the observed months, not as full-season crop stress."
    )

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "FLDAS evapotranspiration (monthly depth; area average)",
        "FLDAS root-zone soil moisture, 10–40 cm (area average)",
        "MODIS MOD13A3 NDVI (nominal 1-km zone samples)",
    ),
)
fig.add_trace(
    go.Scatter(x=water_yearly["SeasonYear"], y=water_yearly["ET_mm_month"],
               mode="lines+markers", name="ET", line={"color": "#c84b31"}),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=water_yearly["SeasonYear"], y=water_yearly["SM_10_40cm"],
               mode="lines+markers", name="Root-zone moisture", line={"color": "#005a9c"}),
    row=2, col=1,
)
for zone, group in ndvi_yearly.groupby("Zone"):
    fig.add_trace(
        go.Scatter(x=group["SeasonYear"], y=group["NDVI"], mode="lines+markers", name=str(zone)),
        row=3, col=1,
    )
fig.update_layout(
    height=760, template="plotly_white", hovermode="x unified",
    legend_title_text="NDVI zone", margin={"t": 65, "b": 35, "l": 25, "r": 20},
)
fig.update_xaxes(title_text="Season year", row=3, col=1)
fig.update_yaxes(title_text="mm/month", row=1, col=1)
fig.update_yaxes(title_text="FLDAS units", row=2, col=1)
fig.update_yaxes(title_text="NDVI", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

st.subheader("How these graphics relate to crop-water stress")
explain_et, explain_moisture, explain_ndvi = st.columns(3)
with explain_et:
    st.markdown("#### 1. Atmospheric water demand")
    st.markdown(
        "ET estimates water transferred from land to the atmosphere. Higher ET can indicate stronger "
        "evaporative demand and, when crops are actively growing, potentially greater water use. ET alone "
        "does **not** show whether crops are stressed: adequate water supply can support high ET, while "
        "severe water limitation can suppress it."
    )
with explain_moisture:
    st.markdown("#### 2. Root-zone water supply")
    st.markdown(
        "FLDAS moisture at 10–40 cm gives regional context for water stored below the surface. A decline "
        "alongside sustained or rising ET may be a reason to investigate possible supply–demand pressure. "
        "It is a modeled area average, not a measurement of water available to a particular crop or zone."
    )
with explain_ndvi:
    st.markdown("#### 3. Vegetation response context")
    st.markdown(
        "NDVI tracks canopy greenness. A coincident NDVI decline can provide a vegetation signal to "
        "investigate alongside water indicators, but it is not specific to water stress: crop stage, harvest, "
        "land cover, pests, and data quality can also affect greenness. The plotted zone samples are nominal "
        "1-km MODIS observations, not field-scale maps."
    )
st.info(
    "**Read the three panels together:** persistent high ET + declining root-zone moisture + declining "
    "NDVI in the same season may flag a period for field checks. These patterns only indicate a possible "
    "crop-water concern; they do not pinpoint or prove crop stress. In this dashboard, FLDAS values are "
    "shared regional averages across all MODIS zones, so cross-zone differences in water stress cannot be inferred."
)

rows = []
for zone, group in paired.groupby("Zone"):
    r_et, p_et = correlation(group["NDVI"], group["ET_mm_month"])
    r_sm, p_sm = correlation(group["NDVI"], group["SM_10_40cm"])
    rows.append({
        "Zone": zone,
        "Paired season-years": int(group["SeasonYear"].nunique()),
        "Mean NDVI": group["NDVI"].mean(),
        "NDVI–ET Pearson r": r_et,
        "NDVI–ET p-value": p_et,
        "NDVI–soil moisture Pearson r": r_sm,
        "NDVI–soil moisture p-value": p_sm,
    })
st.subheader("Zone-level co-variation (exploratory)")
st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
st.caption(
    "Each zone is a representative MODIS sample, while the FLDAS series is a Shakargarh area average "
    "shared across zones—not zone-specific soil moisture or ET. Pearson correlations describe association, "
    "not causation. NDVI measures vegetation greenness, not crop stress or yield; validate signals with "
    "field observations, crop calendars, rainfall, irrigation, and product quality information."
)
with st.expander("How to make a defensible crop-stress claim"):
    st.markdown(
        "A defensible statement is that the dashboard **screens for seasonal co-occurrence** between modeled "
        "water conditions and satellite-observed greenness at representative cropland samples. To claim it "
        "pinpoints crop stress, you would need validated crop-specific stress thresholds or independent field "
        "observations, adequate within-season MODIS coverage, quality screening, and spatially matched water "
        "data for each zone. The shared FLDAS area average in this project does not provide three zone-specific "
        "water measurements."
    )
