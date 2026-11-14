import gradio as gr
import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. DATA LOADER & PREPROCESSOR ---
def load_giovanni_csv(filepath, col_name):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    skip = 0
    for i, line in enumerate(lines):
        if line.lower().startswith('time') or line[:4].isdigit():
            skip = i if line.lower().startswith('time') else i - 1
            break
    df = pd.read_csv(filepath, skiprows=max(0, skip))
    df.columns = ['Date', col_name]
    df['Date'] = pd.to_datetime(df['Date'])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df.dropna()

et = load_giovanni_csv('et.csv', 'ET_raw')
sm_surf = load_giovanni_csv('sm_surface.csv', 'SM_0_10cm')
sm_root = load_giovanni_csv('sm_root.csv', 'SM_10_40cm')

df = et.merge(sm_surf, on='Date').merge(sm_root, on='Date')
df['ET_mm_month'] = df['ET_raw'] * 2592000
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df_complete = df[df['Year'] <= 2025]

# --- 2. INTERACTIVE ANALYSIS ENGINE ---
def analyze_regime(season, split_year):
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

    # Hamed-Rao Modified Mann-Kendall Tests
    mk_et = mk.hamed_rao_modification_test(ann['ET_mm_month'])
    mk_sm = mk.hamed_rao_modification_test(ann['SM_10_40cm'])
    r_val, p_corr = pearsonr(sub_m['SM_10_40cm'], sub_m['ET_mm_month'])

    # Regime Shift Comparison
    early = ann[ann['Year'] < split_year]
    late = ann[ann['Year'] >= split_year]
    et_shift = ((late['ET_mm_month'].mean() - early['ET_mm_month'].mean()) / early['ET_mm_month'].mean()) * 100
    sm_shift = ((late['SM_10_40cm'].mean() - early['SM_10_40cm'].mean()) / early['SM_10_40cm'].mean()) * 100

    # Interactive Plotly Dual-Panel Figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            f"Root-Zone Soil Moisture (10–40 cm) | Sen's Slope: {mk_sm.slope:+.5f}/yr (p = {mk_sm.p:.4f})",
            f"Agricultural Evapotranspiration (mm/mo) | Sen's Slope: {mk_et.slope:+.4f}/yr (p = {mk_et.p:.4f})"
        )
    )

    fig.add_trace(go.Scatter(
        x=ann['Year'], y=ann['SM_10_40cm'], mode='lines+markers',
        name='Root Soil Moisture (m³/m³)', line=dict(color='#0066cc', width=2)
    ), row=1, col=1)

    z_sm = np.polyfit(ann['Year'], ann['SM_10_40cm'], 1)
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=np.poly1d(z_sm)(ann['Year']), mode='lines',
        name='SM Trendline', line=dict(color='#003366', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ann['Year'], y=ann['ET_mm_month'], mode='lines+markers',
        name='Evapotranspiration (mm/mo)', line=dict(color='#d9534f', width=2)
    ), row=2, col=1)

    z_et = np.polyfit(ann['Year'], ann['ET_mm_month'], 1)
    fig.add_trace(go.Scatter(
        x=ann['Year'], y=np.poly1d(z_et)(ann['Year']), mode='lines',
        name='ET Trendline', line=dict(color='#800000', dash='dash')
    ), row=2, col=1)

    fig.add_vline(x=split_year, line_width=1.5, line_dash="dot", line_color="green")
    fig.update_layout(height=540, template="plotly_white", showlegend=False, margin=dict(t=50, b=30, l=50, r=20))

    # Statistical Summary Markdown
    sig_sm = "Statistically Significant (p < 0.05)" if mk_sm.p < 0.05 else "Not Significant (p >= 0.05)"
    sig_et = "Statistically Significant (p < 0.05)" if mk_et.p < 0.05 else "Not Significant (p >= 0.05)"

    summary = f"""
### Hydro-Agricultural Diagnostic: {season}
> **{crop_note}**

| Metric | 44-Yr Trend (Hamed–Rao MK) | Sen's Slope / Yr | p-value | Pre- vs. Post-{split_year} Shift |
| :--- | :--- | :--- | :--- | :--- |
| **Root-Zone Soil Moisture (10–40 cm)** | **{mk_sm.trend.capitalize()}** ({sig_sm}) | `{mk_sm.slope:+.5f}` m³/m³ | `{mk_sm.p:.4f}` | **{sm_shift:+.2f}%** |
| **Evapotranspiration (ET)** | **{mk_et.trend.capitalize()}** ({sig_et}) | `{mk_et.slope:+.4f}` mm/mo | `{mk_et.p:.4f}` | **{et_shift:+.2f}%** |

* **Land–Atmosphere Coupling ($r$):** `{r_val:.3f}` ($p = {p_corr:.2e}$) — confirms how tightly root-zone water availability controls crop water flux in this regime.
"""
    return fig, summary

# --- 3. GRADIO WEB INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="Shakargarh Earth System Trend Detective") as demo:
    gr.Markdown("""
    # 🌍 Earth System Trend Detective: Shakargarh Hydro-Agricultural Observatory
    **NASA 2026 Space Apps Challenge** | Inspired by Prof. Matthew McCabe's Multi-Sensor Agricultural Hydrology Framework  
    *Analyzing 44 years (1982–2026, N = 537 months) of NASA FLDAS root-zone soil moisture and evapotranspiration across the Ravi–Bein agricultural corridor (32.0°N–32.5°N, 75.0°E–75.5°E).*
    """)

    with gr.Row():
        season_dd = gr.Dropdown(
            choices=[
                "Annual Mean (Full Year)",
                "Kharif Season (Jun–Sep: Rice & Monsoon)",
                "Rabi Season (Nov–Apr: Wheat)"
            ],
            value="Annual Mean (Full Year)",
            label="1. Select Cropping / Hydrological Regime"
        )
        split_slider = gr.Slider(
            minimum=1995, maximum=2015, value=2004, step=1,
            label="2. Regime-Shift Split Year (Before vs. After Comparison)"
        )

    plot_out = gr.Plot(label="44-Year Interactive Trend Diagnostics")
    stats_out = gr.Markdown()

    season_dd.change(fn=analyze_regime, inputs=[season_dd, split_slider], outputs=[plot_out, stats_out])
    split_slider.change(fn=analyze_regime, inputs=[season_dd, split_slider], outputs=[plot_out, stats_out])
    demo.load(fn=analyze_regime, inputs=[season_dd, split_slider], outputs=[plot_out, stats_out])

demo.launch(share=True)
