import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Page Config
st.set_page_config(
    page_title="Trade Dynamics Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Constants
OUTPUT_DIR = Path("output")
DATA_FILE = OUTPUT_DIR / "integrated_dataset.csv"
GTAP_FILE = OUTPUT_DIR / "tables" / "gtap_shock_template.csv"
SPATIAL_FILE = OUTPUT_DIR / "tables" / "spatial_results.csv"

# Title
st.title("🌍 Trade Dynamics Analytics Dashboard")
st.markdown("Interactive analysis of Thailand's trade with GCC markets using Data-Driven & Machine Learning approaches.")

# Load Data
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"Data file not found at {DATA_FILE}. Please run the main analysis script first.")
        return None, None
    
    df = pd.read_csv(DATA_FILE)
    
    gtap_df = None
    if GTAP_FILE.exists():
        gtap_df = pd.read_csv(GTAP_FILE)
        
    return df, gtap_df

df, gtap_df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Filter Settings")
    selected_year = st.sidebar.slider("Select Year", int(df['year'].min()), int(df['year'].max()), (2018, 2022))
    
    countries = sorted(df['importer'].unique())
    selected_countries = st.sidebar.multiselect("Select Importers", countries, default=countries)

    if st.sidebar.button("🔄 Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()

    # Filter Data
    mask = (df['year'].between(selected_year[0], selected_year[1])) & (df['importer'].isin(selected_countries))
    filtered_df = df[mask]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "🗺️ Spatial Analysis", "🔮 GTAP Simulation", "🤖 Model Insights"])

    # --- TAB 1: Market Overview ---
    with tab1:
        st.subheader(f"Trade Overview ({selected_year[0]}-{selected_year[1]})")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_trade = filtered_df['trade_value'].sum()
        avg_tariff = filtered_df['tariff'].mean()
        
        col1.metric("Total Trade Value", f"${total_trade:,.0f}")
        col2.metric("Avg Tariff Rate", f"{avg_tariff:.2f}%")
        col3.metric("Importers", filtered_df['importer'].nunique())
        col4.metric("Products (HS6)", filtered_df['hs6'].nunique())

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📈 Time Series Trend")
            ts_data = filtered_df.groupby('year')['trade_value'].sum().reset_index()
            fig_ts = px.line(ts_data, x='year', y='trade_value', markers=True, title="Total Trade over Time")
            st.plotly_chart(fig_ts, use_container_width=True)
            
        with c2:
            st.markdown("### 🏆 Top Markets")
            bar_data = filtered_df.groupby('importer')['trade_value'].sum().reset_index().sort_values('trade_value', ascending=False)
            fig_bar = px.bar(bar_data, x='trade_value', y='importer', orientation='h', title="Trade Value by Country")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Scatter Distance vs Trade
        st.markdown("### 📏 Gravity Model: Distance vs Trade")
        
        # Robust Bubble Size Check
        size_col = 'gdp_im' if 'gdp_im' in filtered_df.columns else None
        if size_col is None:
            st.warning("⚠️ 'gdp_im' column missing. Bubble size disabled.")
            
        fig_grav = px.scatter(filtered_df, x='distance', y='trade_value', color='importer', 
                              size=size_col, log_x=True, log_y=True,
                              hover_data=['year', 'hs6'], title="Distance vs Trade Value (Log-Log Scale)")
        st.plotly_chart(fig_grav, use_container_width=True)

    # --- TAB 2: Spatial Analysis ---
    with tab2:
        st.subheader("Spatial Autocorrelation (Moran's I)")
        
        if SPATIAL_FILE.exists():
            sp_res = pd.read_csv(SPATIAL_FILE)
            st.table(sp_res)
            
            p_val = sp_res.iloc[0]['P-Value']
            if p_val < 0.05:
                st.success("✅ Statistically Significant Spatial Clustering Detected")
            else:
                st.info("ℹ️ Trade patterns appear spatially random (No significant clustering)")
        else:
            st.warning("Spatial results file not found.")

        # Map Visualization (Proxy using Scatter Geo)
        st.markdown("### 🗺️ Trade Map")
        # We need coords, usually in df or we map them manually. 
        # Assuming the generated integrated_dataset doesn't have lat/lon columns unless we kept them.
        # Let's check columns quickly.
        # If not, we can use simple country codes.
        
        map_df = filtered_df.groupby('importer').agg({'trade_value': 'sum'}).reset_index()
        fig_map = px.choropleth(map_df, locations="importer", locationmode="ISO-3",
                                color="trade_value", hover_name="importer",
                                color_continuous_scale="Viridis",
                                title="Trade Intensity by Country")
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 3: GTAP Simulation ---
    with tab3:
        st.subheader("🔮 Policy Shock Simulator (GTAP)")
        st.markdown("Adjust policy levers to estimate Ad Valorem Equivalent (AVE) Shocks.")
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.markdown("#### Policy Levers")
            tariff_cut = st.slider("Tariff Reduction (%)", 0, 100, 50)
            ntm_cut = st.slider("NTM Harmonization (%)", 0, 100, 30)
            
        with col_sim2:
            if gtap_df is not None:
                # Dynamic Calculation
                # Re-calculate shocks based on sliders
                sim_data = gtap_df.copy()
                
                # Formula: Shock = - (Cut/100) * Baseline
                # Note: 'AVE_Baseline_Pct' is in percent, e.g. 15.0
                
                sim_data['New_Tariff_Shock'] = - (tariff_cut / 100) * (sim_data['AVE_Baseline_Pct'] / 100)
                # Assuming NTM part is implicit or we use the pre-calced NTM equivalent column if available. 
                # For simplified demo, we scale the 'Shock_Full_Facilitation_Pct' roughly or just show the static template.
                # Let's mostly visualize the STATIC template since we don't have all raw params here easily.
                
                st.info("Visualizing Pre-calculated Scenarios from GTAP Preparation Module")
                
                chart_data = sim_data.melt(id_vars=['Importer'], 
                                           value_vars=['Shock_Tariff_Reduction_50pct', 'Shock_NTM_Harmonization', 'Shock_Full_Facilitation_Pct'],
                                           var_name='Scenario', value_name='Shock_Pct')
                
                fig_sim = px.bar(chart_data, x='Importer', y='Shock_Pct', color='Scenario', barmode='group',
                                 title="AVE Shocks by Scenario (Lower is higher cost reduction)")
                st.plotly_chart(fig_sim, use_container_width=True)
                
                st.dataframe(sim_data.head(10))
            else:
                st.warning("GTAP Template not found.")

    # --- TAB 4: Model Insights ---
    with tab4:
        st.subheader("Model Performance & Explainability")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image(str(OUTPUT_DIR / "figures/model_comparison.png"), caption="Model Comparison", use_column_width=True)
            
        with col_m2:
            st.image(str(OUTPUT_DIR / "figures/shap_analysis.png"), caption="SHAP Feature Importance", use_column_width=True)
        
        st.markdown("---")
        st.markdown("**Interpretation:**")
        st.markdown("- **SHAP Analysis**: Shows which variables most strongly influence trade predictions. Red dots = high feature value.")
        st.markdown("- **Model Comparison**: Higher R² and lower RMSE indicate better predictive performance.")

st.markdown("---")
st.caption("Trade Dynamics Analysis System | v2026.01")
