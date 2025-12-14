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
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta
import pandas as pd
import folium 
import geopandas as gpd
import zipfile
import os
import tempfile
import gdown

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="GeoSarovar - Water Intelligence", 
    page_icon="💧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING ---
st.markdown("""
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

    h1, h2, h3, .title-font { 
        font-family: 'Rajdhani', sans-serif !important; 
        color: var(--accent-primary) !important; 
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa; 
        border-right: 1px solid #d1d9e6;
    }
    
    /* Module Selector Styling */
    div.row-widget.stRadio > div {
        background-color: #eef2f6;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #d1d9e6;
    }

    /* Primary Buttons */
    div.stButton > button:first-child {
        background: var(--accent-primary);
        border: none;
        color: white !important; 
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem;
        border-radius: 6px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: var(--accent-secondary);
        transform: translateY(-2px);
    }

    /* HUD Header */
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border-bottom: 2px solid var(--accent-primary);
        padding: 15px 25px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    
    /* Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary);
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .alert-card {
        background: #fff5f5;
        border: 1px solid #fc8181;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .date-badge {
        background-color: #eef2f6;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #00204a;
        margin-top: 5px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
try:
    service_account = st.secrets["gcp_service_account"]["client_email"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
    ee.Initialize(credentials)
except Exception:
    try:
        ee.Initialize()
    except Exception as e:
        st.error(f"⚠️ Authentication Error: {e}")
        st.stop()

# --- STATE MANAGEMENT ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = "📍 RWH Site Suitability"

# --- 4. HELPER FUNCTIONS ---

def parse_kml(content):
    try:
        if isinstance(content, bytes): content = content.decode('utf-8')
        match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL | re.IGNORECASE)
        if match: return process_coords(match.group(1))
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.lower().endswith('coordinates') and elem.text:
                return process_coords(elem.text)
    except: pass
    return None

def process_coords(text):
    raw = text.strip().split()
    coords = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in raw if len(x.split(',')) >= 2]
    return ee.Geometry.Polygon([coords]) if len(coords) > 2 else None

@st.cache_data(show_spinner=False)
def load_admin_data(url, is_gdrive=False):
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")
        if is_gdrive:
            gdown.download(url, zip_path, quiet=True, fuzzy=True)
        else:
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(temp_dir)
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    gdf = gpd.read_file(os.path.join(root, file))
                    col_map = {'STATE_UT': 'STATE', 'State': 'STATE', 'Name': 'District', 'Sub_dist': 'Subdistrict'}
                    gdf.rename(columns=col_map, inplace=True)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    return gdf
        return None
    except: return None

def geopandas_to_ee(gdf_row):
    try:
        gjson = json.loads(gdf_row.geometry.to_json())
        geom = gjson['features'][0]['geometry'] if 'features' in gjson else gjson
        return ee.Geometry(geom)
    except: return None

def calculate_area_by_class(image, region, scale):
    area_image = ee.Image.pixelArea().addBands(image)
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='class_index'),
        geometry=region, scale=scale, maxPixels=1e10, bestEffort=True
    )
    groups = stats.get('groups').getInfo()
    data = []
    total_area = 0
    if not groups: return pd.DataFrame()
    for item in groups:
        area_ha = item['sum'] / 10000.0
        total_area += area_ha
        data.append({"Class": f"Class {int(item['class_index'])}", "Area (ha)": area_ha})
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Area (ha)", ascending=False)
        df["%"] = ((df["Area (ha)"] / total_area) * 100).round(1)
        df["Area (ha)"] = df["Area (ha)"].round(2)
    return df

# --- ADVANCED STATIC MAP GENERATOR (FIXED) ---
def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        # 1. Geometry Prep
        if isinstance(roi, ee.Geometry):
            try:
                roi_json = roi.getInfo() 
                roi_bounds = roi.bounds().getInfo()['coordinates'][0]
            except:
                st.error("Error reading ROI geometry.")
                return None
        else:
            roi_json = roi
            roi_bounds = roi['coordinates'][0]

        lons = [p[0] for p in roi_bounds]
        lats = [p[1] for p in roi_bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        mid_lat = (min_lat + max_lat) / 2
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        if height_deg == 0: height_deg = 0.001
        
        aspect_ratio = (width_deg * np.cos(np.radians(mid_lat))) / height_deg
        fig_width = 12 
        fig_height = fig_width / aspect_ratio
        if fig_height > 20: fig_height = 20
        if fig_height < 4: fig_height = 4

        # 2. Prepare Image
        if 'palette' in vis_params or 'min' in vis_params:
            ready_img = image.visualize(**vis_params)
        else:
            ready_img = image 
            
        # 3. Request Thumbnail
        thumb_url = ready_img.getThumbURL({
            'region': roi_json, 
            'dimensions': 1000, 
            'format': 'png', 
            'crs': 'EPSG:4326'
        })
        
        response = requests.get(thumb_url, timeout=120)
        
        if response.status_code != 200:
            st.error(f"GEE Server Error ({response.status_code}): {response.text}")
            return None
            
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            st.error(f"GEE Error: {response.text}")
            return None

        img_pil = Image.open(BytesIO(response.content))
        
        # 4. Plotting
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        extent = [min_lon, max_lon, min_lat, max_lat]
        im = ax.imshow(img_pil, extent=extent, aspect='auto')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#00204a')
        
        ax.tick_params(colors='black', labelcolor='black', labelsize=10)
        ax.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        ax.annotate('N', xy=(0.97, 0.95), xytext=(0.97, 0.88),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', edgecolor='black', width=4, headwidth=12, headlength=10),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        try:
            center_lat = (min_lat + max_lat) / 2
            met_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
            width_met = width_deg * met_per_deg_lon
            target_len_met = width_met / 5
            order = 10 ** np.floor(np.log10(target_len_met))
            nice_len_met = round(target_len_met / order) * order
            nice_len_deg = nice_len_met / met_per_deg_lon
            
            pad_x = width_deg * 0.05
            pad_y = height_deg * 0.05
            start_x = max_lon - pad_x - nice_len_deg
            start_y = min_lat + pad_y
            bar_height = height_deg * 0.015
            
            rect = mpatches.Rectangle((start_x, start_y), nice_len_deg, bar_height, linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
            label = f"{int(nice_len_met/1000)} km" if nice_len_met >= 1000 else f"{int(nice_len_met)} m"
            ax.text(start_x + nice_len_deg/2, start_y + bar_height + (height_deg*0.01), label, color='black', ha='center', va='bottom', fontsize=12, fontweight='bold')
        except: pass
        
        if is_categorical and class_names and 'palette' in vis_params:
            patches = []
            for name, color in zip(class_names, vis_params['palette']):
                patches.append(mpatches.Patch(color=color, label=name))
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False, title="Legend", ncol=min(len(class_names), 4))
            plt.setp(legend.get_title(), color='#00204a', fontweight='bold', fontsize=12)
            for text in legend.get_texts():
                text.set_color("black")
                text.set_fontsize(10)
        elif cmap_colors and 'min' in vis_params:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
            norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color='black')
            cbar.set_label('Index Value', color='black', fontsize=12)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black', fontsize=10)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Map Gen Error: {e}")
        return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/nitesh4004/GeoSarovar/main/geosarovar.png", use_container_width=True)
    st.markdown("### 1. Select Module")
    app_mode = st.radio("Choose Functionality:", ["📍 RWH Site Suitability", "⚠️ Encroachment (S1 SAR)", "🌊 Flood Extent Mapping"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 2. Location (ROI)")
    roi_method = st.radio("Selection Mode", ["Upload KML", "Select Admin Boundary", "Point & Buffer"], label_visibility="collapsed")
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
            with st.spinner("Fetching Data..."):
                gdf = load_admin_data(data_url, is_drive)
            if gdf is not None:
                if 'STATE' in gdf.columns:
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
                    st.info(f"Selected: {len(gdf)} Feature")
    elif roi_method == "Point & Buffer":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", value=20.59)
        lon = c2.number_input("Lon", value=78.96)
        rad = st.number_input("Radius (m)", value=5000)
        new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()

    if new_roi:
        st.session_state['roi'] = new_roi.simplify(maxError=50) 
        st.success("ROI Locked ✅")
    st.markdown("---")
    
    params = {}
    if app_mode == "📍 RWH Site Suitability":
        st.markdown("### 3. Suitability Weights")
        w_rain = st.slider("Rainfall %", 0, 100, 30)
        w_slope = st.slider("Slope %", 0, 100, 20)
        w_lulc = st.slider("Land Cover %", 0, 100, 30)
        w_soil = st.slider("Soil %", 0, 100, 20)
        st.markdown("### 4. Period")
        start = st.date_input("From", datetime.now()-timedelta(365*5))
        end = st.date_input("To", datetime.now())
        params = {'w_rain': w_rain/100, 'w_slope': w_slope/100, 'w_lulc': w_lulc/100, 'w_soil': w_soil/100, 'start': start.strftime("%Y-%m-%d"), 'end': end.strftime("%Y-%m-%d")}

    elif app_mode == "⚠️ Encroachment (S1 SAR)":
        st.markdown("### 3. Comparison Dates")
        orbit = st.radio("Orbit Pass", ["BOTH", "ASCENDING", "DESCENDING"])
        st.markdown("**Initial Period (Baseline)**")
        col1, col2 = st.columns(2)
        d1_start = col1.date_input("Start 1", datetime(2018, 6, 1))
        d1_end = col2.date_input("End 1", datetime(2018, 9, 30))
        st.markdown("**Final Period (Current)**")
        col3, col4 = st.columns(2)
        d2_start = col3.date_input("Start 2", datetime(2024, 6, 1))
        d2_end = col4.date_input("End 2", datetime(2024, 9, 30))
        params = {'d1_start': d1_start.strftime("%Y-%m-%d"), 'd1_end': d1_end.strftime("%Y-%m-%d"), 'd2_start': d2_start.strftime("%Y-%m-%d"), 'd2_end': d2_end.strftime("%Y-%m-%d"), 'orbit': orbit}
    
    elif app_mode == "🌊 Flood Extent Mapping":
        st.markdown("### 3. Flood Event Details")
        orbit = st.radio("Orbit Pass", ["BOTH", "ASCENDING", "DESCENDING"])
        st.markdown("**Before Flood (Dry)**")
        col1, col2 = st.columns(2)
        pre_start = col1.date_input("Pre Start", datetime(2023, 4, 1))
        pre_end = col2.date_input("Pre End", datetime(2023, 6, 1))
        st.markdown("**After Flood (Wet)**")
        col3, col4 = st.columns(2)
        post_start = col3.date_input("Post Start", datetime(2023, 9, 29))
        post_end = col4.date_input("Post End", datetime(2023, 10, 15))
        threshold = st.slider("Difference Threshold", 1.0, 1.5, 1.25, 0.05)
        params = {'pre_start': pre_start.strftime("%Y-%m-%d"), 'pre_end': pre_end.strftime("%Y-%m-%d"), 'post_start': post_start.strftime("%Y-%m-%d"), 'post_end': post_end.strftime("%Y-%m-%d"), 'threshold': threshold, 'orbit': orbit}

    st.markdown("###")
    if st.button("RUN ANALYSIS 🚀"):
        if st.session_state['roi']:
            st.session_state['calculated'] = True
            st.session_state['mode'] = app_mode
            st.session_state['params'] = params
        else:
            st.error("Select ROI first.")

# --- 6. MAIN CONTENT ---
st.markdown(f"""
<div class="hud-header">
    <div>
        <div class="hud-title">GeoSarovar</div>
        <div style="color:#5c6b7f; font-size:0.9rem; font-weight:600;">MODULE: {app_mode.upper()}</div>
    </div>
    <div style="text-align:right;">
        <span style="background:#e6f0ff; color:#00204a; padding:5px 12px; border-radius:20px; font-weight:bold; font-size:0.8rem;">LIVE SYSTEM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper for Safe Map Loading - UPDATED: FORCE HYBRID MODE
def get_safe_map(height=500):
    m = geemap.Map(height=height, basemap="HYBRID")
    return m

if not st.session_state['calculated']:
    st.info("👈 Please select a module and a location in the sidebar to begin.")
    m = get_safe_map(500)
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': 'yellow'}, 'ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    mode = st.session_state['mode']
    p = st.session_state['params']
    col_map, col_res = st.columns([3, 1])
    m = get_safe_map(700)
    m.centerObject(roi, 13)
    image_to_export = None 

    # ==========================================
    # LOGIC A: RWH SITE SUITABILITY
    # ==========================================
    if mode == "📍 RWH Site Suitability":
        with st.spinner("Processing RWH Suitability..."):
            rain = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(p['start'], p['end']).select('precipitation').mean().clip(roi)
            rain_n = rain.clamp(50, 800).unitScale(50, 800)
            dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
            slope = ee.Terrain.slope(dem).clip(roi)
            slope_n = ee.Image(1).subtract(slope.clamp(0, 30).unitScale(0, 30))
            lulc = ee.Image("ESA/WorldCover/v100/2020").select('Map').clip(roi)
            lulc_score = lulc.remap([10, 20, 30, 40, 50, 60, 80], [0.6, 0.8, 0.8, 0.7, 0.0, 1.0, 0.0]).rename('lulc')
            try:
                soil = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').mean().clip(roi)
                soil_n = soil.clamp(0, 50).unitScale(0, 50)
            except: soil_n = ee.Image(0.5).clip(roi)
            
            suitability = (rain_n.multiply(p['w_rain'])).add(slope_n.multiply(p['w_slope'])).add(lulc_score.multiply(p['w_lulc'])).add(soil_n.multiply(p['w_soil']))
            vis = {'min': 0, 'max': 0.8, 'palette': ['d7191c', 'fdae61', 'ffffbf', 'a6d96a', '1a9641']}
            m.addLayer(suitability, vis, 'Suitability Index')
            m.add_colorbar(vis, label="RWH Potential")
            image_to_export = suitability

            try:
                max_val = suitability.reduceRegion(ee.Reducer.max(), roi, 30, maxPixels=1e9).get('constant') 
                if max_val:
                    best_geom = suitability.eq(ee.Number(max_val)).reduceToVectors(geometry=roi, scale=30, geometryType='centroid', maxPixels=1e9)
                    if best_geom.size().getInfo() > 0:
                        pt = best_geom.first().geometry().coordinates().getInfo()
                        folium.Marker([pt[1], pt[0]], popup="Best Site", icon=folium.Icon(color='green', icon='star')).add_to(m)
            except: pass

            with col_res:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">📊 AREA STATS</div>', unsafe_allow_html=True)
                classes = ee.Image(0).where(suitability.lt(0.2), 1).where(suitability.gte(0.2).And(suitability.lt(0.4)), 2)\
                    .where(suitability.gte(0.4).And(suitability.lt(0.6)), 3).where(suitability.gte(0.6).And(suitability.lt(0.8)), 4)\
                    .where(suitability.gte(0.8), 5).clip(roi)
                df = calculate_area_by_class(classes, roi, 30)
                name_map = {"Class 1": "Unsuitable", "Class 2": "Low", "Class 3": "Moderate", "Class 4": "Good", "Class 5": "Excellent"}
                if not df.empty:
                    df['Class'] = df['Class'].map(name_map).fillna(df['Class'])
                    st.dataframe(df, hide_index=True, use_container_width=True)

    # ==========================================
    # LOGIC B: ENCROACHMENT DETECTION (SENTINEL-1)
    # ==========================================
    elif mode == "⚠️ Encroachment (S1 SAR)":
        with st.spinner("Processing Sentinel-1 SAR Data..."):
            
            def get_sar_collection(start_d, end_d, roi_geom, orbit_pass):
                s1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                    .filterDate(start_d, end_d)\
                    .filterBounds(roi_geom)
                if orbit_pass != "BOTH":
                    s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
                return s1

            def process_water_mask(col, roi_geom):
                if col.size().getInfo() == 0: return None, "N/A"
                # Get Date of first image
                date_found = ee.Date(col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                # Speckle Filter
                def speckle_filter(img): return img.select('VV').focal_median(50, 'circle', 'meters').rename('VV_smoothed')
                # Mosaic
                mosaic = col.map(speckle_filter).min().clip(roi_geom)
                # Threshold
                water_mask = mosaic.lt(-16).selfMask()
                return water_mask, date_found

            try:
                col_initial = get_sar_collection(p['d1_start'], p['d1_end'], roi, p['orbit'])
                col_final = get_sar_collection(p['d2_start'], p['d2_end'], roi, p['orbit'])

                water_initial, date_init = process_water_mask(col_initial, roi)
                water_final, date_fin = process_water_mask(col_final, roi)

                if water_initial and water_final:
                    encroachment = water_initial.unmask(0).And(water_final.unmask(0).Not()).selfMask()
                    new_water = water_initial.unmask(0).Not().And(water_final.unmask(0)).selfMask()
                    stable_water = water_initial.unmask(0).And(water_final.unmask(0)).selfMask()

                    change_map = ee.Image(0).where(stable_water, 1).where(encroachment, 2).where(new_water, 3).clip(roi).selfMask()
                    image_to_export = change_map

                    left_layer = geemap.ee_tile_layer(water_initial, {'palette': 'blue'}, "Initial Water")
                    right_layer = geemap.ee_tile_layer(water_final, {'palette': 'cyan'}, "Final Water")
                    m.split_map(left_layer, right_layer)

                    m.addLayer(encroachment, {'palette': 'red'}, '🔴 Encroachment (Loss)')
                    m.addLayer(new_water, {'palette': 'blue'}, '🔵 New Water (Gain)')
                    
                    pixel_area = encroachment.multiply(ee.Image.pixelArea())
                    val_loss = pixel_area.reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e9).values().get(0).getInfo()
                    loss_ha = round((val_loss or 0) / 10000, 2)
                    
                    pixel_area_gain = new_water.multiply(ee.Image.pixelArea())
                    val_gain = pixel_area_gain.reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e9).values().get(0).getInfo()
                    gain_ha = round((val_gain or 0) / 10000, 2)

                    with col_res:
                        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ Change Report")
                        st.metric("🔴 Water Loss", f"{loss_ha} Ha", help="Potential Encroachment")
                        st.metric("🔵 Water Gain", f"{gain_ha} Ha", help="Flooding/New Storage")
                        
                        st.markdown(f"""
                        <div class="date-badge">📅 Base: {date_init}</div>
                        <div class="date-badge">📅 Curr: {date_fin}</div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-label">⏱️ TIMELAPSE</div>', unsafe_allow_html=True)
                        if st.button("Create Timelapse"):
                            with st.spinner("Generating GIF..."):
                                try:
                                    # Reuse collection logic for timelapse
                                    s1_tl = get_sar_collection(p['d1_start'], p['d2_end'], roi, p['orbit']).select('VV')
                                    video_args = {'dimensions': 600, 'region': roi, 'framesPerSecond': 5, 'min': -25, 'max': -5, 'palette': ['black', 'blue', 'white']}
                                    monthly = geemap.create_timeseries(s1_tl, p['d1_start'], p['d2_end'], frequency='year', reducer='median')
                                    gif_url = monthly.getVideoThumbURL(video_args)
                                    st.image(gif_url, caption="Radar Intensity (Dark=Water)", use_container_width=True)
                                except Exception as e: st.error(f"Timelapse Error: {e}")
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Insufficient SAR data for selected dates and orbit.")
                    image_to_export = ee.Image(0)
            except Exception as e:
                st.error(f"Computation Error: {e}")

    # ==========================================
    # LOGIC C: FLOOD EXTENT MAPPING
    # ==========================================
    elif mode == "🌊 Flood Extent Mapping":
        with st.spinner("Processing Flood Extent..."):
            try:
                collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                    .filter(ee.Filter.eq('resolution_meters', 10)) \
                    .filterBounds(roi) \
                    .select('VH')
                
                if p['orbit'] != "BOTH":
                    collection = collection.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))

                before_col = collection.filterDate(p['pre_start'], p['pre_end'])
                after_col = collection.filterDate(p['post_start'], p['post_end'])

                if before_col.size().getInfo() > 0 and after_col.size().getInfo() > 0:
                    # Capture Dates
                    date_pre = ee.Date(before_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    date_post = ee.Date(after_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()

                    before = before_col.median().clip(roi)
                    after = after_col.mosaic().clip(roi)
                    
                    smoothing = 50
                    before_f = before.focal_mean(smoothing, 'circle', 'meters')
                    after_f = after.focal_mean(smoothing, 'circle', 'meters')
                    
                    difference = after_f.divide(before_f)
                    difference_binary = difference.gt(p['threshold'])
                    
                    swater = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('seasonality')
                    swater_mask = swater.gte(10).updateMask(swater.gte(10))
                    flooded_mask = difference_binary.where(swater_mask, 0)
                    flooded = flooded_mask.updateMask(flooded_mask)
                    
                    dem = ee.Image('WWF/HydroSHEDS/03VFDEM')
                    slope = ee.Algorithms.Terrain(dem).select('slope')
                    flooded = flooded.updateMask(slope.lt(5))
                    flooded = flooded.updateMask(flooded.connectedPixelCount().gte(8))
                    
                    image_to_export = flooded

                    m.addLayer(before_f, {'min': -25, 'max': 0}, 'Before Flood (Dry)', False)
                    m.addLayer(after_f, {'min': -25, 'max': 0}, 'After Flood (Wet)', True)
                    m.addLayer(flooded, {'palette': '0000FF'}, '🌊 Estimated Flood Extent')
                    
                    flood_stats = flooded.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, bestEffort=True)
                    flood_area_ha = round(flood_stats.values().get(0).getInfo() / 10000, 2)

                    with col_res:
                        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                        st.markdown("### 🌊 Flood Report")
                        st.metric("Estimated Extent", f"{flood_area_ha} Ha")
                        st.markdown(f"""
                        <div class="date-badge">📅 Pre: {date_pre}</div>
                        <div class="date-badge">📅 Post: {date_post}</div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Orbit: {p['orbit']} | Pol: VH")
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.error(f"No images found for Orbit: {p['orbit']} in these dates.")

            except Exception as e:
                st.error(f"Error: {e}")

    # --- COMMON EXPORT TOOLS ---
    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📥 EXPORTS</div>', unsafe_allow_html=True)
        
        if st.button("Save to Drive (GeoTIFF)"):
            if mode == "📍 RWH Site Suitability": img = suitability
            elif mode == "⚠️ Encroachment (S1 SAR)": img = image_to_export if image_to_export else ee.Image(0)
            elif mode == "🌊 Flood Extent Mapping": img = image_to_export if image_to_export else ee.Image(0)
            
            ee.batch.Export.image.toDrive(
                image=img, description=f"GeoSarovar_Export_{datetime.now().strftime('%Y%m%d')}",
                scale=30, region=roi, folder='GeoSarovar_Exports'
            ).start()
            st.toast("Export started! Check Google Drive.")

        st.markdown("---")
        report_title = st.text_input("Report Title", f"Analysis: {mode}")
        if st.button("Generate Map Image"):
            with st.spinner("Rendering..."):
                img_rep = image_to_export if image_to_export else ee.Image(0)
                vis_rep = {'palette': ['blue']} 
                is_categorical = False
                class_names = None
                cmap = None

                if mode == "📍 RWH Site Suitability": 
                    # Added # for Matplotlib compatibility
                    vis_rep = {'min': 0, 'max': 0.8, 'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']}
                    cmap = vis_rep['palette']
                    is_categorical = False
                elif mode == "⚠️ Encroachment (S1 SAR)": 
                    # Named colors are fine
                    vis_rep = {'min': 1, 'max': 3, 'palette': ['cyan', 'red', 'blue']}
                    is_categorical = True
                    class_names = ['Stable Water', 'Encroachment', 'New Water']
                elif mode == "🌊 Flood Extent Mapping":
                    # Added # for Matplotlib compatibility
                    vis_rep = {'min': 0, 'max': 1, 'palette': ['#0000FF']}
                    is_categorical = True
                    class_names = ['Flood Extent']
                
                buf = generate_static_map_display(img_rep, roi, vis_rep, report_title, cmap_colors=cmap, is_categorical=is_categorical, class_names=class_names)
                if buf:
                    st.download_button("Download JPG", buf, "GeoSarovar_Map.jpg", "image/jpeg", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map:
        m.to_streamlit()
