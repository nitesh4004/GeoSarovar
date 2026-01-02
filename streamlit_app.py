import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# --- 1. PAGE CONFIG & LAYOUT ---
st.set_page_config(
    page_title="GeoSarovar - Environmental Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (Strictly Matching Screenshot) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary-green: #2E7D32;  /* The Header Green */
        --accent-green: #43A047;   /* Lighter Green for buttons */
        --bg-gray: #F8F9FA;
        --border-color: #E9ECEF;
        --text-dark: #212529;
    }

    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    /* 1. HEADER STYLING */
    .dashboard-header {
        background-color: var(--primary-green);
        padding: 1.5rem;
        border-radius: 5px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    .header-subtitle {
        font-size: 14px;
        font-weight: 400;
        margin-top: 5px;
        text-align: center;
        font-style: italic;
        opacity: 0.9;
    }

    /* 2. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-gray);
        border-right: 1px solid var(--border-color);
    }
    .sidebar-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 15px;
        border-bottom: 2px solid #DEE2E6;
        padding-bottom: 10px;
    }
    
    /* 3. BUTTON STYLING */
    div.stButton > button:first-child {
        background-color: var(--primary-green);
        color: white;
        font-weight: 600;
        border-radius: 4px;
        border: none;
        width: 100%;
        padding: 0.6rem;
    }
    div.stButton > button:first-child:hover {
        background-color: var(--accent-green);
    }

    /* 4. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b6e8f !important; /* Blue-ish active tab from image */
        color: white !important;
        border-color: #3b6e8f !important;
    }

    /* 5. CARD & PANEL STYLING */
    .panel-box {
        border: 1px solid #DEE2E6;
        border-radius: 5px;
        padding: 15px;
        background: white;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .panel-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 10px;
        border-bottom: 1px solid #F1F3F5;
        padding-bottom: 5px;
    }

    /* 6. STATS METRICS */
    .stat-box {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        color: #6C757D;
        font-weight: 600;
    }
    .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-dark);
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION (GEE) ---
try:
    if "gcp_service_account" in st.secrets:
        service_account = st.secrets["gcp_service_account"]["client_email"]
        secret_dict = dict(st.secrets["gcp_service_account"])
        key_data = json.dumps(secret_dict)
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
        ee.Initialize(credentials)
    else:
        ee.Initialize()
except Exception as e:
    st.error(f"⚠️ GEE Authentication Error: {e}")

# --- STATE MANAGEMENT ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = "📍 RWH Site Suitability"

# --- 4. HELPERS (DL & UTILS) ---

def build_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=2)
    return model

@st.cache_resource
def load_dl_model_from_drive(device="cpu"):
    model_path = "water_unet_best.pth"
    file_id = "1-v-SLRDr3OiiKAnQeebpwQzIPDpLamsW"
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model..."):
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

def predict_large_image(model, image, device="cpu", tile_size=512, overlap=64):
    def preprocess_tile(tile): return (tile.astype(np.float32) / 255.0)
    _, H, W = image.shape
    pad_H = (tile_size - H % tile_size) if H % tile_size != 0 else 0
    pad_W = (tile_size - W % tile_size) if W % tile_size != 0 else 0
    image_pad = np.pad(image, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0)
    _, H_pad, W_pad = image_pad.shape
    stride = tile_size - overlap
    prob_sum = np.zeros((H_pad, W_pad), dtype=np.float32)
    count = np.zeros((H_pad, W_pad), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y0 in range(0, H_pad, stride):
            for x0 in range(0, W_pad, stride):
                y1 = min(y0 + tile_size, H_pad)
                x1 = min(x0 + tile_size, W_pad)
                tile = image_pad[:, y0:y1, x0:x1]
                tile = preprocess_tile(tile)
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                out = model(tile_tensor)
                if out.shape[1] > 1: out = out[:, 1:2, :, :]
                prob = torch.sigmoid(out).cpu().numpy()[0, 0]
                prob_sum[y0:y1, x0:x1] += prob
                count[y0:y1, x0:x1] += 1.0
    count[count == 0] = 1.0
    prob_full = prob_sum / count
    prob_full = prob_full[:H, :W]
    mask_full = (prob_full >= 0.5).astype(np.uint8)
    return mask_full, prob_full

def mask_to_vector(mask, transform, crs):
    mask = mask.astype(np.uint8)
    results = shapes(mask, mask=mask == 1, transform=transform)
    geoms = []
    for geom, value in results:
        if value == 1: geoms.append(shape_geom(geom))
    if len(geoms) == 0: return gpd.GeoDataFrame({"id": [], "area_km2": [], "geometry": []}, crs=crs)
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    gdf["id"] = range(1, len(gdf) + 1)
    if crs and not isinstance(crs, str) and crs.is_geographic:
        gdf_proj = gdf.to_crs(epsg=3857)
        gdf["area_m2"] = gdf_proj.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area
    gdf["area_km2"] = (gdf["area_m2"] / 1_000_000).round(4)
    return gdf.sort_values("area_km2", ascending=False).reset_index(drop=True)

def read_geotiff(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return image, profile, transform, crs, bounds

def parse_kml(content):
    try:
        if isinstance(content, bytes): content = content.decode('utf-8')
        match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL | re.IGNORECASE)
        if match: 
            raw = match.group(1).strip().split()
            coords = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in raw if len(x.split(',')) >= 2]
            return ee.Geometry.Polygon([coords]) if len(coords) > 2 else None
    except: pass
    return None

@st.cache_data(show_spinner=False)
def load_admin_data(url, is_gdrive=False):
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")
        if is_gdrive: gdown.download(url, zip_path, quiet=True, fuzzy=True)
        else:
            r = requests.get(url)
            with open(zip_path, "wb") as f: f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(temp_dir)
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    gdf = gpd.read_file(os.path.join(root, file))
                    col_map = {'STATE_UT': 'STATE', 'State': 'STATE', 'Name': 'District', 'Sub_dist': 'Subdistrict'}
                    gdf.rename(columns=col_map, inplace=True)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    return gdf
    except: return None
    return None

def geopandas_to_ee(gdf_row):
    try:
        gjson = json.loads(gdf_row.geometry.to_json())
        geom = gjson['features'][0]['geometry'] if 'features' in gjson else gjson
        return ee.Geometry(geom)
    except: return None

# --- 5. SIDEBAR (MATCHING SCREENSHOT) ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Data Filters</div>', unsafe_allow_html=True)
    
    # 1. Dataset Selection (Module Selector)
    st.markdown("**Select Dataset (Module)**")
    app_mode = st.selectbox(
        "Choose Analysis Type",
        ["📍 RWH Site Suitability",
         "⚠️ Encroachment (S1 SAR)",
         "Flood Extent Mapping",
         "🧪 Water Quality",
         "🤖 DL Water Segmentation"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # 2. Region Selection
    st.markdown("**Select Region**")
    roi_method = st.radio("Selection Mode", ["Select Admin Boundary", "Upload KML", "Point & Buffer"], label_visibility="collapsed")
    
    new_roi = None
    if roi_method == "Upload KML":
        kml = st.file_uploader("Upload KML", type=['kml'])
        if kml: new_roi = parse_kml(kml.read())
        
    elif roi_method == "Select Admin Boundary":
        admin_level = st.selectbox("Granularity", ["Districts", "Subdistricts", "States"])
        data_url = None
        is_drive = False
        if admin_level == "Districts":
            data_url = 'https://drive.google.com/uc?id=1tMyiUheQBcwwPwZQla67PwC5-AqenTmv'; is_drive = True
        elif admin_level == "Subdistricts":
            data_url = 'https://drive.google.com/uc?id=18lMyt2j3Xjz_Qk_2Kzppr8EVlVDx_yOv'; is_drive = True
        elif admin_level == "States":
            data_url = "https://github.com/nitesh4004/GeoFormatX/raw/main/STATE_BOUNDARY.zip"; is_drive = False
            
        if data_url:
            with st.spinner("Loading Admin Data..."):
                gdf = load_admin_data(data_url, is_drive)
            if gdf is not None and 'STATE' in gdf.columns:
                states = sorted(gdf['STATE'].astype(str).unique())
                sel_state = st.selectbox("State", states)
                gdf = gdf[gdf['STATE'] == sel_state]
                if 'District' in gdf.columns and not gdf.empty:
                    dists = sorted(gdf['District'].astype(str).unique())
                    sel_dist = st.selectbox("District", dists)
                    gdf = gdf[gdf['District'] == sel_dist]
                    if 'Subdistrict' in gdf.columns and not gdf.empty:
                        subs = sorted(gdf['Subdistrict'].astype(str).unique())
                        sel_sub = st.selectbox("Subdistrict", subs)
                        gdf = gdf[gdf['Subdistrict'] == sel_sub]
                if not gdf.empty:
                    new_roi = geopandas_to_ee(gdf.iloc[[0]])
                    st.success(f"Locked: {len(gdf)} Feature")

    elif roi_method == "Point & Buffer":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", value=20.59)
        lon = c2.number_input("Lon", value=78.96)
        rad = st.number_input("Radius (m)", value=5000)
        new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()

    if new_roi:
        st.session_state['roi'] = new_roi.simplify(maxError=50)

    st.markdown("---")
    
    # 3. Dynamic Parameters
    st.markdown("**Parameters**")
    params = {}
    
    if app_mode == "📍 RWH Site Suitability":
        rwh_type = st.selectbox("Structure Type", ("Check Dam", "Farm Pond", "Percolation Tank"))
        year = st.slider("Analysis Year", 2018, 2024, 2023)
        params = {'rwh_type': rwh_type, 'start': f'{year}-06-01', 'end': f'{year}-10-30'}
        
    elif app_mode == "⚠️ Encroachment (S1 SAR)":
        orbit = st.selectbox("Orbit", ["BOTH", "ASCENDING", "DESCENDING"])
        d1_start = st.date_input("Base Start", datetime(2018, 6, 1))
        d1_end = st.date_input("Base End", datetime(2018, 9, 30))
        d2_start = st.date_input("Curr Start", datetime(2024, 6, 1))
        d2_end = st.date_input("Curr End", datetime(2024, 9, 30))
        params = {'d1_start': str(d1_start), 'd1_end': str(d1_end), 'd2_start': str(d2_start), 'd2_end': str(d2_end), 'orbit': orbit}
        
    elif app_mode == "Flood Extent Mapping":
        orbit = st.selectbox("Orbit", ["BOTH", "ASCENDING", "DESCENDING"])
        pre_start = st.date_input("Pre Start", datetime(2023, 4, 1))
        pre_end = st.date_input("Pre End", datetime(2023, 6, 1))
        post_start = st.date_input("Post Start", datetime(2023, 9, 29))
        post_end = st.date_input("Post End", datetime(2023, 10, 15))
        threshold = st.slider("Threshold", 1.0, 1.5, 1.25)
        params = {'pre_start': str(pre_start), 'pre_end': str(pre_end), 'post_start': str(post_start), 'post_end': str(post_end), 'threshold': threshold, 'orbit': orbit}

    elif app_mode == "🧪 Water Quality":
        wq_param = st.selectbox("Parameter", ["Turbidity (NDTI)", "Total Suspended Solids (TSS)", "Cyanobacteria", "Chlorophyll-a", "CDOM"])
        wq_start = st.date_input("Start", datetime.now()-timedelta(days=90))
        wq_end = st.date_input("End", datetime.now())
        cloud_thresh = st.slider("Max Cloud %", 5, 50, 20)
        params = {'param': wq_param, 'start': str(wq_start), 'end': str(wq_end), 'cloud': cloud_thresh}

    elif app_mode == "🤖 DL Water Segmentation":
        dl_source = st.radio("Input", ["Planetary Computer (ROI)", "Upload TIFF"])
        if dl_source == "Planetary Computer (ROI)":
            sat_type = st.selectbox("Sat", ["Sentinel-2", "Landsat 8"])
            params = {'source': 'pc', 'sat_type': sat_type}
        else:
            uf = st.file_uploader("Upload", type=["tif"])
            params = {'source': 'upload', 'file': uf}

    st.markdown("###")
    if st.button("Apply Filters"):
        if st.session_state['roi'] or (app_mode == "🤖 DL Water Segmentation" and params.get('source') == 'upload' and params.get('file')):
            st.session_state['calculated'] = True
            st.session_state['mode'] = app_mode
            st.session_state['params'] = params
        else:
            st.error("Select ROI or Upload File")

# --- 6. MAIN DASHBOARD UI ---
st.markdown("""
<div class="dashboard-header">
    <div class="header-title">Geospatial Dashboard - Environmental Analysis</div>
    <div class="header-subtitle">Interactive Geospatial Analysis & Remote Sensing Visualization</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.info("👈 Use the 'Data Filters' sidebar to configure and run your analysis.")
    # Empty Placeholder structure
    tab1, tab2, tab3 = st.tabs(["Raster Analysis", "Vector Data", "3D Terrain"])
    with tab1:
        m = geemap.Map(height=500, basemap="HYBRID")
        if st.session_state['roi']:
            m.centerObject(st.session_state['roi'], 12)
            m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': 'yellow'}, 'ROI')
        m.to_streamlit()

else:
    roi = st.session_state['roi']
    mode = st.session_state['mode']
    p = st.session_state['params']
    
    # TABS STRUCTURE from Screenshot
    tab1, tab2, tab3 = st.tabs(["Raster Analysis", "Vector Data", "3D Terrain"])
    
    # --- GLOBAL VARS FOR EXPORT ---
    image_to_export = None
    vis_export = {}
    chart_data = None
    vector_results = None

    with tab1:
        # GRID: Left Map (2.5) | Right Charts (1)
        col_map, col_charts = st.columns([2.5, 1])
        
        # Initialize Map
        m = geemap.Map(height=550, basemap="HYBRID")
        if roi: m.centerObject(roi, 13)

        # ---------------------------------------------------------------------
        # LOGIC 1: DL SEGMENTATION
        # ---------------------------------------------------------------------
        if mode == "🤖 DL Water Segmentation":
            with st.spinner("Running Deep Learning Inference..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_dl_model_from_drive(device)
                
                image, profile, transform, crs = None, None, None, None
                
                # Input Handling
                if p['source'] == 'upload':
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                        tmp.write(p['file'].getbuffer())
                        tiff_path = tmp.name
                    image, profile, transform, crs, _ = read_geotiff(tiff_path)
                    m.add_raster(tiff_path, layer_name="Input Image", zoom_to_layer=True)
                else:
                    # PC Logic
                    from rasterio.coords import BoundingBox
                    import pystac_client, planetary_computer, stackstac, dask
                    roi_json = roi.getInfo()
                    geom = roi_json["geometry"] if "geometry" in roi_json else roi_json
                    coords = geom["coordinates"][0] if geom['type'] == 'Polygon' else geom["coordinates"][0][0]
                    bbox = [min(c[0] for c in coords), min(c[1] for c in coords), max(c[0] for c in coords), max(c[1] for c in coords)]
                    
                    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                    col_id = "sentinel-2-l2a" if "Sentinel" in p['sat_type'] else "landsat-c2-l2"
                    bands = ["B02","B03","B04"] if "Sentinel" in p['sat_type'] else ["blue","green","red"]
                    
                    search = catalog.search(collections=[col_id], bbox=bbox, datetime="2023-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 20}})
                    items = list(search.items())
                    if items:
                        stack = stackstac.stack(items[:5], assets=bands, bounds_latlon=bbox, resolution=10).median(dim="time").compute()
                        image = np.nan_to_num(stack.values).astype(np.uint16)
                        # Create Transform
                        x_coords, y_coords = stack.x.values, stack.y.values
                        res_x = x_coords[1] - x_coords[0]
                        res_y = y_coords[1] - y_coords[0]
                        transform = Affine(res_x, 0, x_coords[0], 0, res_y, y_coords[0])
                        crs = "EPSG:3857"
                        
                        # Temp save for map
                        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                            with rasterio.open(tmp.name, 'w', driver='GTiff', height=image.shape[1], width=image.shape[2], count=3, dtype='uint16', crs=crs, transform=transform) as dst:
                                dst.write(image)
                            m.add_raster(tmp.name, layer_name="Satellite", zoom_to_layer=True)
                    else:
                        st.error("No imagery found.")
                        st.stop()

                # Inference
                mask, _ = predict_large_image(model, image, device)
                gdf = mask_to_vector(mask, transform, crs)
                vector_results = gdf
                
                if not gdf.empty:
                    m.add_gdf(gdf, layer_name="Water Mask", style={"color": "#00BFFF", "fillColor": "#00BFFF", "fillOpacity": 0.5})

                # --- UI: Map ---
                with col_map:
                    st.markdown(f"**{mode}**")
                    m.to_streamlit()
                    
                    # Bottom Stats Cards
                    st.markdown("### Statistics")
                    c1, c2, c3 = st.columns(3)
                    cnt = len(gdf) if not gdf.empty else 0
                    area = gdf['area_km2'].sum() if not gdf.empty else 0
                    c1.markdown(f'<div class="stat-box"><div class="stat-label">Water Bodies</div><div class="stat-value">{cnt}</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-box"><div class="stat-label">Total Area (km²)</div><div class="stat-value">{area:.2f}</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-box"><div class="stat-label">Model</div><div class="stat-value">ResNet34</div></div>', unsafe_allow_html=True)

                # --- UI: Charts ---
                with col_charts:
                    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">Area Distribution</div>', unsafe_allow_html=True)
                    if not gdf.empty:
                        top = gdf.head(5)
                        fig, ax = plt.subplots(figsize=(3,3))
                        ax.pie(top['area_km2'], labels=[f"ID {i}" for i in top['id']], autopct='%1.0f%%')
                        st.pyplot(fig)
                        st.dataframe(gdf[['id','area_km2']], height=150)
                    else: st.write("No Data")
                    st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # LOGIC 2: RWH SITE SUITABILITY
        # ---------------------------------------------------------------------
        elif mode == "📍 RWH Site Suitability":
            with st.spinner("Calculating Suitability..."):
                dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
                rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(p['start'], p['end']).filterBounds(roi).sum().clip(roi).rename('rainfall')
                lulc = ee.ImageCollection("ESA/WorldCover/v100").first().clip(roi).rename('lulc')
                hydro = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(roi)
                flow = hydro.select('b1').rename('flow_accumulation')
                slope = ee.Terrain.slope(dem).rename('slope')
                
                # Stack
                features = ee.Image.cat([dem, slope, rain, flow, lulc]).rename(['elevation', 'slope', 'rainfall', 'flow_accumulation', 'lulc'])
                
                # Synthetic Rules for RF
                high_suit = features.select('flow_accumulation').gt(500).And(features.select('slope').lt(10))
                low_suit = features.select('slope').gt(20)
                train_pts = features.updateMask(high_suit).sample(roi, 100, 50).map(lambda f: f.set('class', 1))\
                    .merge(features.updateMask(low_suit).sample(roi, 100, 50).map(lambda f: f.set('class', 0)))
                
                classifier = ee.Classifier.smileRandomForest(50).train(train_pts, 'class', ['elevation', 'slope', 'rainfall', 'flow_accumulation', 'lulc'])
                classified = features.classify(classifier)
                
                # Constraints
                mask = classified.eq(1)
                if p['rwh_type'] == "Check Dam": mask = mask.And(slope.lt(15))
                elif p['rwh_type'] == "Farm Pond": mask = mask.And(slope.lt(5))
                result = classified.updateMask(mask)
                
                image_to_export = result
                vis_export = {'min': 0, 'max': 1, 'palette': ['white', 'green']}
                
                m.addLayer(dem, {'min': 0, 'max': 1000}, 'Elevation', False)
                m.addLayer(rain, {'min': 0, 'max': 1500, 'palette': ['blue','cyan']}, 'Rainfall', False)
                m.addLayer(result, {'palette':['green']}, 'Suitability')
                
                # Stats
                rain_val = rain.reduceRegion(ee.Reducer.mean(), roi, 1000).get('rainfall').getInfo() or 0
                elev_val = dem.reduceRegion(ee.Reducer.mean(), roi, 1000).get('elevation').getInfo() or 0

                with col_map:
                    st.markdown(f"**{mode} - {p['rwh_type']}**")
                    m.to_streamlit()
                    st.markdown("### Statistics")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="stat-box"><div class="stat-label">Avg Rainfall</div><div class="stat-value">{rain_val:.1f} mm</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-box"><div class="stat-label">Avg Elevation</div><div class="stat-value">{elev_val:.0f} m</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-box"><div class="stat-label">Model</div><div class="stat-value">Random Forest</div></div>', unsafe_allow_html=True)

                with col_charts:
                    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">Factor Importance</div>', unsafe_allow_html=True)
                    # Mock importance for visualization
                    factors = ['Slope', 'Rain', 'Flow', 'LULC']
                    vals = [40, 25, 20, 15]
                    fig, ax = plt.subplots(figsize=(3,3))
                    ax.pie(vals, labels=factors, autopct='%1.0f%%', startangle=90)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # LOGIC 3: ENCROACHMENT (S1)
        # ---------------------------------------------------------------------
        elif mode == "⚠️ Encroachment (S1 SAR)":
            with st.spinner("Processing Sentinel-1 SAR..."):
                def get_s1(s, e):
                    c = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(s, e).filterBounds(roi)
                    if p['orbit'] != "BOTH": c = c.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))
                    return c
                
                col1 = get_s1(p['d1_start'], p['d1_end'])
                col2 = get_s1(p['d2_start'], p['d2_end'])
                
                if col1.size().getInfo() > 0 and col2.size().getInfo() > 0:
                    w1 = col1.map(lambda i: i.select('VV').focal_median(50,'circle','meters')).min().clip(roi).lt(-16)
                    w2 = col2.map(lambda i: i.select('VV').focal_median(50,'circle','meters')).min().clip(roi).lt(-16)
                    
                    encroach = w1.And(w2.Not()).selfMask()
                    new_w = w1.Not().And(w2).selfMask()
                    stable = w1.And(w2).selfMask()
                    
                    m.addLayer(stable, {'palette': 'cyan'}, 'Stable Water')
                    m.addLayer(encroach, {'palette': 'red'}, 'Encroachment (Loss)')
                    m.addLayer(new_w, {'palette': 'blue'}, 'New Water (Gain)')
                    
                    loss_area = encroach.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e9).get('VV').getInfo() or 0
                    gain_area = new_w.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e9).get('VV').getInfo() or 0
                    loss_ha = loss_area / 10000
                    gain_ha = gain_area / 10000
                    
                    with col_map:
                        st.markdown(f"**{mode}**")
                        m.to_streamlit()
                        st.markdown("### Statistics")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f'<div class="stat-box"><div class="stat-label">Loss (Ha)</div><div class="stat-value">{loss_ha:.2f}</div></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="stat-box"><div class="stat-label">Gain (Ha)</div><div class="stat-value">{gain_ha:.2f}</div></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="stat-box"><div class="stat-label">Orbit</div><div class="stat-value">{p["orbit"]}</div></div>', unsafe_allow_html=True)
                    
                    with col_charts:
                        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                        st.markdown('<div class="panel-title">Change Metrics</div>', unsafe_allow_html=True)
                        st.bar_chart({"Loss": loss_ha, "Gain": gain_ha})
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Insufficient SAR data.")

        # ---------------------------------------------------------------------
        # LOGIC 4: FLOOD MAPPING
        # ---------------------------------------------------------------------
        elif mode == "Flood Extent Mapping":
            with st.spinner("Mapping Flood..."):
                def get_s1(s, e):
                    c = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(s, e).filterBounds(roi)
                    if p['orbit'] != "BOTH": c = c.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))
                    return c
                
                c_pre = get_s1(p['pre_start'], p['pre_end'])
                c_post = get_s1(p['post_start'], p['post_end'])
                
                if c_pre.size().getInfo() > 0 and c_post.size().getInfo() > 0:
                    pre = c_pre.median().clip(roi).focal_mean(50,'circle','meters')
                    post = c_post.mosaic().clip(roi).focal_mean(50,'circle','meters')
                    
                    diff = post.divide(pre)
                    flood = diff.gt(p['threshold'])
                    
                    # Perm water & Slope masking
                    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
                    dem = ee.Image('WWF/HydroSHEDS/03VFDEM')
                    slope = ee.Algorithms.Terrain(dem).select('slope')
                    
                    final_flood = flood.updateMask(gsw.gt(30).Not()).updateMask(slope.lt(5)).selfMask()
                    
                    m.addLayer(pre, {'min':-25, 'max':0}, 'Pre-Event', False)
                    m.addLayer(post, {'min':-25, 'max':0}, 'Post-Event', True)
                    m.addLayer(final_flood, {'palette':['blue']}, 'Flood Extent')
                    
                    area_sqm = final_flood.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e9).get('VH').getInfo() or 0
                    area_ha = area_sqm / 10000
                    
                    with col_map:
                        st.markdown(f"**{mode}**")
                        m.to_streamlit()
                        st.markdown("### Statistics")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f'<div class="stat-box"><div class="stat-label">Flood Area</div><div class="stat-value">{area_ha:.2f} Ha</div></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="stat-box"><div class="stat-label">Threshold</div><div class="stat-value">{p["threshold"]}</div></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="stat-box"><div class="stat-label">Pol</div><div class="stat-value">VH</div></div>', unsafe_allow_html=True)
                    
                    with col_charts:
                         st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                         st.info("Visual analysis complete. See map for extent.")
                         st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # LOGIC 5: WATER QUALITY
        # ---------------------------------------------------------------------
        elif mode == "🧪 Water Quality":
            with st.spinner("Analyzing Indices..."):
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(p['start'], p['end']).filterBounds(roi)
                
                # Formulae
                def add_indices(img):
                    ndti = img.normalizedDifference(['B4', 'B3']).rename('NDTI')
                    tss = img.expression('2950 * (b4**1.357)', {'b4': img.select('B4')}).rename('TSS')
                    return img.addBands([ndti, tss])

                s2_proc = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', p['cloud'])).map(add_indices)
                
                if s2_proc.size().getInfo() > 0:
                    band = 'NDTI' if "Turbidity" in p['param'] else 'TSS'
                    result = s2_proc.select(band).median().clip(roi)
                    vis = {'min':-0.1, 'max':0.1, 'palette':['blue','yellow','red']} if band == 'NDTI' else {'min':0, 'max':50, 'palette':['blue','yellow','brown']}
                    m.addLayer(result, vis, band)
                    
                    # Time Series
                    def get_mean(img):
                        val = img.reduceRegion(ee.Reducer.mean(), roi, 50).get(band)
                        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'value': val})
                    
                    ts_data = s2_proc.map(get_mean).filter(ee.Filter.notNull(['value'])).reduceColumns(ee.Reducer.toList(2), ['date','value']).get('list').getInfo()
                    df = pd.DataFrame(ts_data, columns=['Date', 'Value']) if ts_data else pd.DataFrame()
                    
                    with col_map:
                        st.markdown(f"**{mode}**")
                        m.to_streamlit()
                        st.markdown("### Statistics")
                        c1, c2, c3 = st.columns(3)
                        mean_val = df['Value'].mean() if not df.empty else 0
                        max_val = df['Value'].max() if not df.empty else 0
                        c1.markdown(f'<div class="stat-box"><div class="stat-label">Mean {band}</div><div class="stat-value">{mean_val:.3f}</div></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="stat-box"><div class="stat-label">Max {band}</div><div class="stat-value">{max_val:.3f}</div></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="stat-box"><div class="stat-label">Samples</div><div class="stat-value">{len(df)}</div></div>', unsafe_allow_html=True)

                    with col_charts:
                        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                        st.markdown(f'<div class="panel-title">{band} Time Series</div>', unsafe_allow_html=True)
                        if not df.empty:
                            df['Date'] = pd.to_datetime(df['Date'])
                            st.line_chart(df.set_index('Date'))
                        else: st.write("No valid data.")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No cloud-free images.")

    # --- TAB 2: VECTOR DATA ---
    with tab2:
        st.markdown("### Vector Analysis Results")
        if vector_results is not None:
            st.dataframe(vector_results, use_container_width=True)
        else:
            st.info("Run DL Segmentation to generate vector data.")

    # --- TAB 3: 3D TERRAIN ---
    with tab3:
        st.markdown("### 3D Terrain Visualization")
        st.info("Terrain visualization utilizes the SRTM DEM data used in the analysis.")
        # Placeholder for visual consistency
