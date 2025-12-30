import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import numpy as np
import pandas as pd
import folium
import geopandas as gpd
import os
import torch
import segmentation_models_pytorch as smp
from shapely.geometry import shape as shape_geom
from rasterio.features import shapes

# -------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GeoSarovar - Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 2. CSS STYLING
# -------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;600&display=swap');

    :root {
        --bg-color: #ffffff;
        --accent-primary: #00204a;
        --accent-secondary: #005792;
        --text-primary: #00204a;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        color: var(--accent-primary) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #d1d9e6;
    }

    div.stButton > button:first-child {
        background: var(--accent-primary);
        border: none;
        color: white !important;
        font-weight: 700;
        padding: 0.6rem;
        border-radius: 6px;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        background: var(--accent-secondary);
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00204a;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# 3. GEE AUTHENTICATION
# -------------------------------------------------
try:
    # Check if we are in a Streamlit Cloud env with secrets
    if "gcp_service_account" in st.secrets:
        service_account = st.secrets["gcp_service_account"]["client_email"]
        key_data = json.dumps(dict(st.secrets["gcp_service_account"]))
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
        ee.Initialize(credentials)
    else:
        # Local authentication
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
except Exception as e:
    st.error(f"GEE Authentication Error: {e}")
    st.stop()

# -------------------------------------------------
# 4. SESSION STATE & HELPER FUNCTIONS
# -------------------------------------------------
if "calculated" not in st.session_state:
    st.session_state["calculated"] = False
if "roi" not in st.session_state:
    st.session_state["roi"] = None
if "layers" not in st.session_state:
    st.session_state["layers"] = []

def get_safe_map(height=600):
    return geemap.Map(height=height, basemap="HYBRID")

# -------------------------------------------------
# 5. SIDEBAR & INPUTS
# -------------------------------------------------
st.sidebar.title("GeoSarovar Control")
st.sidebar.markdown("---")

# Module Selection
mode = st.sidebar.radio(
    "Select Module",
    ["📍 RWH Site Suitability", "💧 Water Body Extraction (DL)"]
)
st.session_state["mode"] = mode

# --- RWH MODULE INPUTS ---
if mode == "📍 RWH Site Suitability":
    st.sidebar.subheader("1. Area of Interest")
    lat = st.sidebar.number_input("Center Latitude", value=18.5204)
    lon = st.sidebar.number_input("Center Longitude", value=73.8567)
    
    st.sidebar.subheader("2. Structure Parameters")
    rwh_type = st.sidebar.selectbox(
        "Structure Type",
        ("Check Dam", "Farm Pond", "Percolation Tank")
    )
    
    st.sidebar.subheader("3. Rainfall Scenario")
    year = st.sidebar.slider("Analysis Year", 2018, 2024, 2023)
    
    # Action Button
    run_btn = st.sidebar.button("Run Suitability Analysis")

# --- DL MODULE INPUTS (Placeholder for existing DL code) ---
else:
    st.sidebar.info("Deep Learning Module selected. Upload imagery to begin.")
    # (Existing DL inputs would go here)
    run_btn = False

# -------------------------------------------------
# 6. MAIN APPLICATION LOGIC
# -------------------------------------------------
st.title("GeoSarovar – Water Intelligence Platform")

# -----------------------------------------------------------------------------
# LOGIC BLOCK: RWH SITE SUITABILITY
# -----------------------------------------------------------------------------
if mode == "📍 RWH Site Suitability":
    
    if run_btn:
        st.session_state["calculated"] = True
        
        with st.spinner("Acquiring Satellite Data (SRTM, CHIRPS, Sentinel-2)..."):
            # 1. Define ROI (Buffer point)
            roi = ee.Geometry.Point([lon, lat]).buffer(15000) # 15km radius
            st.session_state["roi"] = roi
            
            # 2. Data Acquisition
            start_date = f'{year}-06-01'
            end_date = f'{year}-10-30'
            
            # DEM & Slope
            dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
            slope = ee.Terrain.slope(dem).rename('slope')
            
            # Rainfall (CHIRPS)
            rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                .filterDate(start_date, end_date) \
                .filterBounds(roi) \
                .sum().clip(roi).rename('rainfall')
                
            # Soil (OpenLandMap)
            soil = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1a1a_M/v02") \
                .select('b0').clip(roi).rename('soil_sand')
                
            # LULC (WorldCover)
            lulc = ee.ImageCollection("ESA/WorldCover/v100").first().clip(roi).rename('lulc')
            
            # Hydro (Flow Accumulation)
            hydro = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(roi)
            flow_acc = hydro.select('b1').rename('flow_accumulation')

        with st.spinner("Engineering Features & Running Random Forest..."):
            # 3. Feature Engineering & Stacking
            # Normalize inputs implicitly by RF, but we stack them for training
            features = ee.Image.cat([dem, slope, rainfall, flow_acc, soil, lulc])
            feature_names = ['elevation', 'slope', 'rainfall', 'flow_accumulation', 'soil_sand', 'lulc']
            features = features.rename(feature_names)
            
            # 4. Constraints (Step 5 of Methodology)
            # Remove urban (50), steep slopes (>30 deg)
            mask_slope = slope.lt(30)
            mask_urban = lulc.neq(50)
            
            # Specific structure constraints
            if rwh_type == "Check Dam":
                mask_struct = slope.lt(15) 
            elif rwh_type == "Farm Pond":
                mask_struct = slope.lt(5)
            else:
                mask_struct = ee.Image(1)
                
            final_mask = mask_slope.And(mask_urban).And(mask_struct)
            processed_features = features.updateMask(final_mask)
            
            # 5. Synthetic Training Data (Since no ground truth CSV provided)
            # Good: High Flow Acc + Low Slope
            # Bad: Steep Slope OR Low Rainfall
            high_suitability_rule = flow_acc.gt(500).And(slope.lt(10))
            low_suitability_rule = slope.gt(20)
            
            points_good = processed_features.updateMask(high_suitability_rule).sample(
                region=roi, scale=100, numPixels=50, geometries=True
            ).map(lambda f: f.set('class', 1))
            
            points_bad = processed_features.updateMask(low_suitability_rule).sample(
                region=roi, scale=100, numPixels=50, geometries=True
            ).map(lambda f: f.set('class', 0))
            
            training_data = points_good.merge(points_bad)
            
            # 6. Random Forest Modeling
            classifier = ee.Classifier.smileRandomForest(numberOfTrees=50) \
                .train(training_data, 'class', feature_names)
                
            suitability_map = processed_features.classify(classifier)
            
            # Store result in session state to persist map
            st.session_state["rwh_result"] = {
                "dem": dem,
                "slope": slope,
                "suitability": suitability_map,
                "rainfall": rainfall
            }

    # --- DISPLAY OUTPUTS ---
    if st.session_state.get("calculated") and "rwh_result" in st.session_state:
        res = st.session_state["rwh_result"]
        
        # Layout: Map on top, Stats below
        st.subheader(f"Results: {rwh_type} Suitability ({year})")
        
        # Map Visualization
        m = get_safe_map()
        m.centerObject(st.session_state["roi"], 11)
        
        # Visual Params
        vis_suitability = {'min': 0, 'max': 1, 'palette': ['#ff4d4d', '#00cc66']} # Red to Green
        vis_slope = {'min': 0, 'max': 30, 'palette': ['black', 'white']}
        
        m.addLayer(res['dem'], {'min': 0, 'max': 1000}, 'Elevation (DEM)', False)
        m.addLayer(res['slope'], vis_slope, 'Slope', False)
        m.addLayer(res['rainfall'], {'min': 0, 'max': 2000, 'palette': ['blue', 'cyan']}, 'Rainfall', False)
        m.addLayer(res['suitability'], vis_suitability, 'RWH Suitability (AI)', True)
        
        m.to_streamlit(height=600)
        
        # Decision Support Analytics
        st.markdown("### 📊 Decision Support Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <b>Topographic Feasibility</b><br>
                Evaluated using SRTM DEM.
                <br><i>Steep slopes excluded.</i>
            </div>
            """, unsafe_allow_html=True)
            st.caption("")
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <b>Hydrological Potential</b><br>
                Based on Flow Accumulation.
                <br><i>Catchment areas identified.</i>
            </div>
            """, unsafe_allow_html=True)
            st.caption("")
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <b>AI Confidence</b><br>
                Random Forest Classifier.
                <br><i>50 Trees, 6 Features.</i>
            </div>
            """, unsafe_allow_html=True)
            st.caption("

[Image of random forest algorithm diagram]
")

# -----------------------------------------------------------------------------
# LOGIC BLOCK: DEFAULT / WELCOME SCREEN
# -----------------------------------------------------------------------------
elif not st.session_state["calculated"]:
    st.info("👈 Select a module from the sidebar and click 'Run' to begin.")
    
    # Default map view
    m = get_safe_map()
    m.setCenter(78.9629, 20.5937, 5) # India View
    m.to_streamlit()

    st.markdown("""
    ### About GeoSarovar
    GeoSarovar is an advanced hydrological intelligence platform combining **Satellite Remote Sensing** and **Artificial Intelligence**.
    
    **Available Modules:**
    1.  **RWH Site Suitability:** Uses Random Forest (ML) on GEE to identify optimal locations for Check Dams, Farm Ponds, etc.
    2.  **Water Body Extraction:** Uses U-Net (Deep Learning) to segment water bodies from high-resolution imagery.
    """)
