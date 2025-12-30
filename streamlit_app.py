import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd
import folium
import geopandas as gpd
import zipfile
import os
import tempfile
import gdown
import torch
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape as shape_geom
import segmentation_models_pytorch as smp
from rasterio.transform import Affine

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
    if "gcp_service_account" in st.secrets:
        service_account = st.secrets["gcp_service_account"]["client_email"]
        key_data = json.dumps(dict(st.secrets["gcp_service_account"]))
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
        ee.Initialize(credentials)
    else:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
except Exception as e:
    st.error(f"GEE Authentication Error: {e}")
    st.stop()

# -------------------------------------------------
# 4. SESSION STATE
# -------------------------------------------------
st.session_state.setdefault("calculated", False)
st.session_state.setdefault("roi", None)
st.session_state.setdefault("mode", "📍 RWH Site Suitability")
st.session_state.setdefault("dl_result", None)

# -------------------------------------------------
# 5. DL MODEL HELPERS
# -------------------------------------------------
def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    )
    return model

@st.cache_resource
def load_dl_model_from_drive(device="cpu"):
    model_path = "water_unet_best.pth"
    file_id = "1-v-SLRDr3OiiKAnQeebpwQzIPDpLamsW"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model weights..."):
            gdown.download(url, model_path, quiet=False)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model = build_model()
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model

def preprocess_tile(tile):
    tile = tile.astype(np.float32) / 255.0
    return tile

def predict_large_image(model, image, device="cpu", tile_size=512, overlap=64):
    _, h, w = image.shape
    pad_h = (tile_size - h % tile_size) if h % tile_size != 0 else 0
    pad_w = (tile_size - w % tile_size) if w % tile_size != 0 else 0

    image_pad = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    stride = tile_size - overlap

    prob_sum = np.zeros(image_pad.shape[1:], dtype=np.float32)
    count = np.zeros(image_pad.shape[1:], dtype=np.float32)

    with torch.no_grad():
        for y in range(0, image_pad.shape[1], stride):
            for x in range(0, image_pad.shape[2], stride):
                tile = image_pad[:, y:y+tile_size, x:x+tile_size]
                tile = preprocess_tile(tile)
                tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)

                out = model(tile_t)
                out = torch.sigmoid(out[:, 1:2, :, :])
                prob = out.cpu().numpy()[0, 0]

                prob_sum[y:y+tile_size, x:x+tile_size] += prob
                count[y:y+tile_size, x:x+tile_size] += 1

    prob = prob_sum / np.maximum(count, 1)
    mask = (prob[:h, :w] >= 0.5).astype(np.uint8)
    return mask, prob[:h, :w]

def mask_to_vector(mask, transform, crs):
    results = shapes(mask, mask=mask == 1, transform=transform)
    geoms = [shape_geom(geom) for geom, val in results if val == 1]

    if not geoms:
        return gpd.GeoDataFrame(columns=["id", "area_km2", "geometry"], crs=crs)

    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    gdf["id"] = range(1, len(gdf) + 1)

    gdf_proj = gdf.to_crs(epsg=3857)
    gdf["area_km2"] = (gdf_proj.area / 1_000_000).round(4)

    return gdf.sort_values("area_km2", ascending=False).reset_index(drop=True)

def get_safe_map(height=600):
    return geemap.Map(height=height, basemap="HYBRID")

# -------------------------------------------------
# 6. UI ENTRY POINT & SIDEBAR
# -------------------------------------------------
st.sidebar.title("GeoSarovar Control")
st.sidebar.markdown("---")

# Module Selection
mode = st.sidebar.radio(
    "Select Module",
    ["📍 RWH Site Suitability", "💧 Water Body Extraction (DL)"]
)
st.session_state["mode"] = mode

# -----------------
# 6.1 RWH INPUTS
# -----------------
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
    
    run_btn = st.sidebar.button("Run Suitability Analysis")

# -----------------
# 6.2 DL INPUTS
# -----------------
else:
    st.sidebar.subheader("Deep Learning Input")
    uploaded_file = st.sidebar.file_uploader("Upload Satellite Image (Tiff)", type=["tif", "tiff"])
    run_btn = st.sidebar.button("Extract Water Bodies")

# -------------------------------------------------
# 7. MAIN APPLICATION LOGIC
# -------------------------------------------------
st.title("GeoSarovar – Water Intelligence Platform")

# -------------------------------------------------------
# LOGIC A: RWH SITE SUITABILITY (GEE + RANDOM FOREST)
# -------------------------------------------------------
if mode == "📍 RWH Site Suitability":
    
    if run_btn:
        st.session_state["calculated"] = True
        
        with st.spinner("Acquiring Satellite Data (SRTM, CHIRPS, Sentinel-2)..."):
            # 1. Define ROI
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
            # 3. Feature Engineering
            features = ee.Image.cat([dem, slope, rainfall, flow_acc, soil, lulc])
            feature_names = ['elevation', 'slope', 'rainfall', 'flow_accumulation', 'soil_sand', 'lulc']
            features = features.rename(feature_names)
            
            # 4. Constraints
            mask_slope = slope.lt(30)
            mask_urban = lulc.neq(50)
            
            if rwh_type == "Check Dam":
                mask_struct = slope.lt(15) 
            elif rwh_type == "Farm Pond":
                mask_struct = slope.lt(5)
            else:
                mask_struct = ee.Image(1)
                
            final_mask = mask_slope.And(mask_urban).And(mask_struct)
            processed_features = features.updateMask(final_mask)
            
            # 5. Synthetic Training Data (Heuristic based)
            high_suitability_rule = flow_acc.gt(500).And(slope.lt(10))
            low_suitability_rule = slope.gt(20)
            
            points_good = processed_features.updateMask(high_suitability_rule).sample(
                region=roi, scale=100, numPixels=50, geometries=True
            ).map(lambda f: f.set('class', 1))
            
            points_bad = processed_features.updateMask(low_suitability_rule).sample(
                region=roi, scale=100, numPixels=50, geometries=True
            ).map(lambda f: f.set('class', 0))
            
            training_data = points_good.merge(points_bad)
            
            # 6. Modeling
            classifier = ee.Classifier.smileRandomForest(numberOfTrees=50) \
                .train(training_data, 'class', feature_names)
                
            suitability_map = processed_features.classify(classifier)
            
            st.session_state["rwh_result"] = {
                "dem": dem,
                "slope": slope,
                "suitability": suitability_map,
                "rainfall": rainfall
            }

    # Display RWH Output
    if st.session_state.get("calculated") and "rwh_result" in st.session_state:
        res = st.session_state["rwh_result"]
        
        st.subheader(f"Results: {rwh_type} Suitability ({year})")
        
        m = get_safe_map()
        m.centerObject(st.session_state["roi"], 11)
        
        vis_suitability = {'min': 0, 'max': 1, 'palette': ['#ff4d4d', '#00cc66']}
        vis_slope = {'min': 0, 'max': 30, 'palette': ['black', 'white']}
        
        m.addLayer(res['dem'], {'min': 0, 'max': 1000}, 'Elevation (DEM)', False)
        m.addLayer(res['slope'], vis_slope, 'Slope', False)
        m.addLayer(res['rainfall'], {'min': 0, 'max': 2000, 'palette': ['blue', 'cyan']}, 'Rainfall', False)
        m.addLayer(res['suitability'], vis_suitability, 'RWH Suitability (AI)', True)
        
        m.to_streamlit(height=600)
        
        st.markdown("### 📊 Decision Support Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card"><b>Topographic Feasibility</b><br>Evaluated using SRTM DEM. </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card"><b>Hydrological Potential</b><br>Based on Flow Accumulation. </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card"><b>AI Confidence</b><br>Random Forest Classifier. 

[Image of random forest algorithm diagram]
</div>""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOGIC B: WATER BODY EXTRACTION (DEEP LEARNING)
# -------------------------------------------------------
elif mode == "💧 Water Body Extraction (DL)":
    
    if run_btn and uploaded_file is not None:
        st.session_state["calculated"] = True
        
        # Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_dl_model_from_drive(device)
        
        # Process Image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        with rasterio.open(tmp_path) as src:
            image = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs

        # Ensure 6 bands (padding if necessary, or error handling)
        if image.shape[0] != 6:
            st.warning(f"Model expects 6 bands, but got {image.shape[0]}. Attempting to proceed (results may vary).")
            # In production, add logic to handle different band counts
            
        # Prediction
        with st.spinner("Running U-Net Segmentation..."):
            image_transposed = image  # Shape is already (C, H, W) for PyTorch
            mask, prob_map = predict_large_image(model, image_transposed, device=device)
            
        # Vectorize
        gdf = mask_to_vector(mask, transform, crs)
        
        # Display DL Output
        st.subheader("Deep Learning Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(prob_map, caption="Probability Heatmap", clamp=True, use_column_width=True)
        
        with col2:
            st.write(f"Detected {len(gdf)} water bodies.")
            st.dataframe(gdf[["id", "area_km2"]].head())
            
            # Download Logic
            csv = gdf.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "water_bodies.csv", "text/csv")

# -------------------------------------------------------
# DEFAULT / WELCOME SCREEN
# -------------------------------------------------------
if not st.session_state["calculated"]:
    st.info("👈 Select a module from the sidebar and click 'Run' to begin.")
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
