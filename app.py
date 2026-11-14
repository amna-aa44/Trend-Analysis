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


def trend_label(result):
    return result.trend.capitalize() if result is not None else "Not enough data/variation"


def trend_p(result):
    return f"{result.p:.4f}" if result is not None else "N/A"


def trend_slope(result, unit):
    return f"{result.slope:+.5g} {unit}" if result is not None else "N/A"


def trend_interpretation(result, indicator, unit):
    if result is None:
        return f"There is insufficient variation or yearly data to estimate a reliable trend in {indicator}."
    evidence = "statistically significant" if result.p < 0.05 else "not statistically significant"
    return (
        f"The estimated direction is **{result.trend.lower()}**. Sen's slope is "
        f"**{result.slope:+.5g} {unit} per year**, and the p-value is **{result.p:.4f}**; "
        f"this trend is {evidence} at the 0.05 threshold."
    )


def comparison_direction(value):
    if pd.isna(value):
        return "cannot be compared because one period has no usable data"
    if value > 0:
        return f"increased by {value:.2f}%"
    if value < 0:
        return f"decreased by {abs(value):.2f}%"
    return "was effectively unchanged"


st.title("🌾 The story of Shakargarh's resources: Crop-Water stress in the past 4 decades")
st.markdown(
    """### Research overview
    This dashboard examines long-term seasonal variability in modeled evapotranspiration (ET) and
    root-zone soil moisture across Shakargarh using NASA FLDAS monthly area-average time series.
    Where available, Terra MODIS NDVI provides a complementary satellite-derived measure of vegetation
    greenness. Comparing ET and soil moisture characterizes their temporal association and seasonal
    co-variation; the trend tests assess monotonic changes in each indicator over time. Results are
    descriptive, and do not by themselves establish causation, crop-yield impacts, or field-level
    water availability."""
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

with st.expander("Methods and data interpretation"):
    st.markdown(
        """- **ET:** Giovanni monthly mean flux converted to monthly depth using each month's actual duration.
        - **Root-zone moisture:** FLDAS soil moisture for the 10–40 cm layer.
        - **Relative demand index:** ET divided by root-zone moisture scaled by 300. It is an exploratory ratio, not a validated drought or crop-stress index.
        - **Trend test:** Hamed–Rao modified Mann–Kendall test applied to yearly/seasonal averages. Statistical significance does not establish cause.
        - **Scope:** FLDAS values are modeled area averages. They should guide questions and monitoring priorities, not replace field measurements, irrigation records, rainfall data, or agronomic advice."""
    )

st.subheader("📈 Seasonal water trends")
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

# First explanatory board: association, trend method, Sen slope and p-values.
with st.container(border=True):
    st.subheader("Relation between Evapotranspiration and Soil Moisture")
    st.markdown(
        f"For **{season}**, the monthly paired observations show a Pearson correlation of "
        f"**r = {r_value:.3f}** (p = **{p_value:.3g}**). This describes the strength and direction "
        "of linear co-variation in the selected months; it does not show that one variable causes the other."
        if pd.notna(r_value)
        else "There are not enough paired observations with variation to calculate a reliable monthly correlation for this selection."
    )

    relation_chart_col, relation_stats_col = st.columns([1.25, 1], gap="large")
    with relation_chart_col:
        scatter_data = selected[["SM_10_40cm", "ET_mm_month"]].dropna()
        if len(scatter_data) >= 2:
            scatter = go.Figure()
            scatter.add_trace(
                go.Scatter(
                    x=scatter_data["SM_10_40cm"],
                    y=scatter_data["ET_mm_month"],
                    mode="markers",
                    marker={"color": "#167d8d", "opacity": 0.65, "size": 8},
                    name="Monthly observations",
                    customdata=selected.loc[scatter_data.index, "Date"].astype(str),
                    hovertemplate="Date: %{customdata}<br>Soil moisture: %{x:.4f}<br>ET: %{y:.3f} mm/month<extra></extra>",
                )
            )
            if scatter_data["SM_10_40cm"].nunique() > 1:
                slope_line, intercept_line = np.polyfit(
                    scatter_data["SM_10_40cm"], scatter_data["ET_mm_month"], 1
                )
                x_line = np.linspace(scatter_data["SM_10_40cm"].min(), scatter_data["SM_10_40cm"].max(), 100)
                scatter.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=slope_line * x_line + intercept_line,
                        mode="lines",
                        line={"color": "#c84b31", "dash": "dash"},
                        name="Linear guide",
                    )
                )
            scatter.update_layout(
                template="plotly_white",
                xaxis_title="Root-zone soil moisture (10–40 cm)",
                yaxis_title="Evapotranspiration (mm/month)",
                height=390,
                margin={"t": 25, "b": 45, "l": 45, "r": 15},
            )
            st.plotly_chart(scatter, use_container_width=True)
        else:
            st.info("Not enough monthly pairs are available to draw the relationship plot.")

    with relation_stats_col:
        st.markdown("#### Annual/seasonal Mann–Kendall results")
        st.caption("The test is applied separately to the annual/seasonal ET and soil-moisture series.")
        stat_rows = []
        for name, result, unit in [
            ("Evapotranspiration", mk_et, "mm/month per year"),
            ("Root-zone soil moisture", mk_sm, "soil-moisture units per year"),
        ]:
            stat_rows.append(
                {
                    "Indicator": name,
                    "Trend": trend_label(result),
                    "Sen slope": trend_slope(result, unit),
                    "p-value": trend_p(result),
                }
            )
        st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

        with st.expander("How to interpret the Mann–Kendall test, slope, and p-value"):
            st.markdown(
                """- **Mann–Kendall test:** A non-parametric test for whether a series has a consistent monotonic upward or downward trend over time. The Hamed–Rao modification adjusts the test for serial autocorrelation.
                - **Sen slope:** The median rate of change between observations. Positive means the indicator tends to increase; negative means it tends to decrease. Units are shown for each indicator.
                - **p-value:** Under the no-monotonic-trend null hypothesis and test assumptions, this is the probability of observing a test statistic at least this extreme. A common threshold is **p < 0.05**; it is not the probability that the trend is true.
                - **Important distinction:** Pearson r summarizes monthly ET–moisture association; the Mann–Kendall test and Sen slope assess long-run trends in each indicator separately."""
            )
        st.markdown("#### Trend statements for this selection")
        st.markdown(
            f"**ET:** {trend_interpretation(mk_et, 'evapotranspiration', 'mm/month')}\n\n"
            f"**Soil moisture:** {trend_interpretation(mk_sm, 'root-zone soil moisture', 'moisture units')}"
        )

# Second explanatory board: use actual values and allow the reader to choose their perspective.
with st.container(border=True):
    st.subheader("What does it mean for you?")
    audience = st.selectbox(
        "Choose a perspective to see how to use this result",
        ["Farmer or farm manager", "Researcher or student", "Extension worker or planner", "General reader"],
        key="audience_perspective",
    )

    et_direction = comparison_direction(et_change)
    moisture_direction = comparison_direction(sm_change)
    stress_direction = comparison_direction(stress_change)

    st.markdown(
        f"In **{season}**, comparing the period before **{split_year}** with {split_year} onward, "
        f"the mean ET {et_direction}, while root-zone soil moisture {moisture_direction}. "
        f"The exploratory relative-demand index {stress_direction}."
    )

    if audience == "Farmer or farm manager":
        st.markdown(
            "Use this regional seasonal signal as a prompt for **when and where to check fields**, not as an irrigation prescription. "
            "If modeled ET rises while modeled root-zone moisture falls or remains stable, prioritize field soil-moisture checks and crop scouting during the selected months. "
            "Before changing irrigation, compare with recent rainfall, soil type, crop stage, and on-farm observations."
        )
    elif audience == "Researcher or student":
        st.markdown(
            "Use the selected seasonal series to formulate hypotheses about water demand and root-zone supply. "
            "Report the Mann–Kendall direction, Sen slope, p-value, sample period, and the monthly Pearson association separately. "
            "For attribution, validate against precipitation, temperature, irrigation records, crop calendars, and field measurements; the dashboard alone cannot identify causes."
        )
    elif audience == "Extension worker or planner":
        st.markdown(
            "Use the result to identify seasons or locations where additional monitoring may be valuable. "
            "Pair regional FLDAS patterns with local soil-moisture observations, rainfall records, and farmer reports before developing outreach or water-management recommendations."
        )
    else:
        st.markdown(
            "The comparison summarizes how modeled water use and root-zone moisture changed between two periods. "
            "It is useful as regional context, but it does not directly measure crop yields, irrigation demand, or the conditions of a particular field."
        )

    with st.expander("Read the result responsibly"):
        st.markdown(
            "A period-to-period percentage change is descriptive, while the trend test evaluates a monotonic pattern across the full selected time series. "
            "A significant trend is not proof of cause. The FLDAS data are modeled and spatially averaged; check local observations before making field-level decisions. "
            "The relative-demand index is an exploratory ratio, not a validated crop-stress threshold."
        )

st.subheader("Additional trend evidence")
trend_rows = []
for name, result, unit in [
    ("Evapotranspiration", mk_et, "mm/month per year"),
    ("Root-zone soil moisture", mk_sm, "moisture units per year"),
    ("Relative demand index", mk_stress, "index units per year"),
]:
    trend_rows.append(
        {
            "Indicator": name,
            "Direction": trend_label(result),
            "Sen slope": trend_slope(result, unit),
            "p-value": trend_p(result),
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
                    "Trend": trend_label(result),
                    "Sen slope per year": f"{result.slope:+.5f}" if result else "N/A",
                    "p-value": trend_p(result),
                }
            )
        ndvi_chart.update_layout(
            height=430, template="plotly_white", yaxis_title="NDVI",
            xaxis_title="Year", hovermode="x unified",
        )
        st.plotly_chart(ndvi_chart, use_container_width=True)
        st.dataframe(pd.DataFrame(ndvi_rows), hide_index=True, use_container_width=True)
        st.caption(
            "NDVI is a satellite-derived vegetation greenness measure, not a direct measure of crop yield. "
            "Crop stage and sensor effects should be considered when interpreting changes."
        )

st.caption(
    f"Sources: NASA Giovanni FLDAS monthly area-average time series and Terra MODIS MOD13A3 NDVI where available. "
    f"Analysis includes observations through {MAX_YEAR}."
)
