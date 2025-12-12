import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import folium 
import geopandas as gpd
import zipfile
import os
import tempfile
import gdown

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="GeoSarovar - RWH Analytics", 
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
        --card-bg: #ffffff;
        --border-color: #d1d9e6;
        --accent-primary: #00204a;   /* Dark Navy Blue */
        --accent-secondary: #005792; /* Brighter Blue */
        --text-primary: #00204a;     /* Dark Blue Text */
        --text-secondary: #5c6b7f;
    }

    .stApp { 
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    h1, h2, h3, .title-font { 
        font-family: 'Rajdhani', sans-serif !important; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        color: var(--accent-primary) !important; 
    }
    
    p, label, .stMarkdown, div, span { 
        color: var(--text-primary) !important; 
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa; /* Very Light Grey */
        border-right: 1px solid var(--border-color);
    }
    
    /* Input Fields */
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        background-color: #ffffff !important;
        border: 1px solid #00204a !important; /* Dark Blue Border */
        border-radius: 6px;
        color: #00204a !important;
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
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 32, 74, 0.2);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        background: var(--accent-secondary);
        box-shadow: 0 6px 12px rgba(0, 87, 146, 0.3);
    }
    div.stButton > button:first-child p {
        color: white !important;
    }

    /* HUD Header */
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border-bottom: 2px solid var(--accent-primary);
        padding: 20px 25px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    .hud-badge {
        background: #e6f0ff;
        border: 1px solid var(--accent-primary);
        color: var(--accent-primary);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }

    /* Clean White Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary);
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 8px;
    }
    
    iframe {
        border-radius: 12px;
        border: 2px solid #00204a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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

# --- SESSION STATE INITIALIZATION ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None

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

def calculate_area_by_class(image, region, scale):
    area_image = ee.Image.pixelArea().addBands(image)
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='class_index'),
        geometry=region,
        scale=scale,
        maxPixels=1e10, 
        bestEffort=True
    )
    
    groups = stats.get('groups').getInfo()
    data = []
    total_area = 0
    
    if not groups: return pd.DataFrame()

    for item in groups:
        c_idx = int(item['class_index'])
        area_sqm = item['sum']
        area_ha = area_sqm / 10000.0
        total_area += area_ha
        data.append({"Class": f"Class {c_idx}", "Area (ha)": area_ha})
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Area (ha)", ascending=False)
        df["%"] = ((df["Area (ha)"] / total_area) * 100).round(1)
        df["Area (ha)"] = df["Area (ha)"].round(2)
        
    return df

def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        roi_bounds = roi.bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in roi_bounds]
        lats = [p[1] for p in roi_bounds]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        mid_lat = (min_lat + max_lat) / 2
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        aspect_ratio = (width_deg * np.cos(np.radians(mid_lat))) / height_deg
        fig_width = 12 
        fig_height = fig_width / aspect_ratio
        if fig_height > 20: fig_height = 20
        if fig_height < 4: fig_height = 4

        if 'palette' in vis_params or 'min' in vis_params:
            ready_img = image.visualize(**vis_params)
        else:
            ready_img = image 
            
        thumb_url = ready_img.getThumbURL({
            'region': roi,
            'dimensions': 1500, 
            'format': 'png',
            'crs': 'EPSG:4326' 
        })
        response = requests.get(thumb_url, timeout=120)
        
        if response.status_code != 200: return None
        img_pil = Image.open(BytesIO(response.content))
        
        # White Theme Plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        im = ax.imshow(img_pil, extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#00204a')
        ax.tick_params(colors='#00204a', labelcolor='#00204a', labelsize=10)
        ax.grid(color='#00204a', linestyle='--', linewidth=0.5, alpha=0.15)
        for spine in ax.spines.values():
            spine.set_edgecolor('#00204a')
            spine.set_alpha(0.5)
        
        # North Arrow
        ax.annotate('N', xy=(0.97, 0.95), xytext=(0.97, 0.88),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='#00204a', edgecolor='white', width=4, headwidth=12, headlength=10),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='#00204a',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")])

        # Legend logic
        if is_categorical and class_names and 'palette' in vis_params:
            patches = []
            for name, color in zip(class_names, vis_params['palette']):
                patches.append(mpatches.Patch(color=color, label=name))
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                               frameon=False, title="Classes", ncol=min(len(class_names), 4))
            plt.setp(legend.get_title(), color='#00204a', fontweight='bold', fontsize=12)
            for text in legend.get_texts():
                text.set_color("#00204a")
                
        elif cmap_colors and 'min' in vis_params:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
            norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color='#00204a')
            cbar.set_label('Value', color='#00204a', fontsize=12)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#00204a', fontsize=10)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

# --- ADMIN DATA LOADER (FIXED WITH GDOWN) ---
@st.cache_data(show_spinner=False)
def load_admin_data(url, is_gdrive=False):
    """
    Downloads and reads shapefile. 
    Uses gdown for Google Drive links to handle permissions/large files.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")
        
        if is_gdrive:
            # Gdown handles drive links much better than requests
            gdown.download(url, zip_path, quiet=True, fuzzy=True)
        else:
            # Standard request for direct links (Github)
            response = requests.get(url, stream=True)
            if response.status_code != 200: return None
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract and Find Shapefile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    gdf = gpd.read_file(os.path.join(root, file))
                    
                    # Standardization of Column Names
                    col_map = {
                        'STATE_UT': 'STATE', 'State': 'STATE',
                        'Name': 'District', 'Sub_dist': 'Subdistrict'
                    }
                    gdf.rename(columns=col_map, inplace=True)
                    
                    # Convert Text to String to avoid object errors
                    for col in ['District', 'STATE', 'Subdistrict']:
                        if col in gdf.columns:
                            gdf[col] = gdf[col].astype(str).str.strip()

                    # Ensure EPSG:4326 for Earth Engine
                    if gdf.crs is None:
                        gdf.set_crs(epsg=4326, inplace=True)
                    elif gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                        
                    return gdf
        return None
    except Exception as e:
        return None

def geopandas_to_ee(gdf_row):
    """Converts a single GeoPandas row to Earth Engine Geometry"""
    try:
        # Get GeoJSON geometry from the row
        gjson = json.loads(gdf_row.geometry.to_json())
        # Extract coordinates and type
        if 'features' in gjson:
            geom_data = gjson['features'][0]['geometry']
        else:
            geom_data = gjson
        return ee.Geometry(geom_data)
    except Exception as e:
        st.error(f"Geometry conversion error: {e}")
        return None

# --- 5. SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    # LOGO DISPLAY
    st.image("https://raw.githubusercontent.com/nitesh4004/GeoSarovar/main/geosarovar.png", use_container_width=True)
    
    st.markdown("""
        <div style="margin-bottom: 20px; text-align: center;">
            <p style="font-size: 0.85rem; color: #00204a; letter-spacing: 2px; margin-top:5px; font-weight:700;">INTELLIGENT RWH ANALYTICS</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 1. Site Selection (ROI)")
        
        # REORDERED OPTIONS HERE as requested
        roi_method = st.radio(
            "Selection Mode", 
            ["Upload KML", "Select Admin Boundary", "Point & Buffer", "Manual Coordinates"], 
            label_visibility="collapsed"
        )
        
        new_roi = None

        # --- OPTION 1: ADMIN BOUNDARY (Uses GDOWN for Drive Links) ---
        if roi_method == "Select Admin Boundary":
            admin_level = st.selectbox("Granularity", ["Districts", "Subdistricts", "States"])
            
            # URL Mapping
            data_url = None
            is_drive = False
            
            if admin_level == "Districts":
                data_url = 'https://drive.google.com/uc?id=1tMyiUheQBcwwPwZQla67PwC5-AqenTmv'
                is_drive = True
            elif admin_level == "Subdistricts":
                data_url = 'https://drive.google.com/uc?id=18lMyt2j3Xjz_Qk_2Kzppr8EVlVDx_yOv'
                is_drive = True
            elif admin_level == "States":
                data_url = "https://github.com/nitesh4004/GeoFormatX/raw/main/STATE_BOUNDARY.zip"
                is_drive = False

            if data_url:
                with st.spinner("Fetching Administrative Data..."):
                    gdf = load_admin_data(data_url, is_drive)
                
                if gdf is not None:
                    # Filter Logic
                    final_selection = gdf
                    
                    if 'STATE' in gdf.columns:
                        states = sorted(gdf['STATE'].astype(str).unique())
                        sel_state = st.selectbox("State", states)
                        final_selection = final_selection[final_selection['STATE'] == sel_state]
                        
                        if 'District' in gdf.columns and not final_selection.empty:
                            dists = sorted(final_selection['District'].astype(str).unique())
                            sel_dist = st.selectbox("District", dists)
                            final_selection = final_selection[final_selection['District'] == sel_dist]
                            
                            if 'Subdistrict' in gdf.columns and not final_selection.empty:
                                subs = sorted(final_selection['Subdistrict'].astype(str).unique())
                                sel_sub = st.selectbox("Subdistrict", subs)
                                final_selection = final_selection[final_selection['Subdistrict'] == sel_sub]
                    
                    # Convert Result to EE
                    if not final_selection.empty:
                        # Take the first match
                        row = final_selection.iloc[[0]] 
                        st.info(f"Selected: {len(final_selection)} Feature(s)")
                        new_roi = geopandas_to_ee(row)
                else:
                    st.error("Failed to load map data. Check internet or Drive permissions.")
        
        elif roi_method == "Upload KML":
            kml = st.file_uploader("Drop KML File", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            c1, c2 = st.columns([1, 1])
            lat = c1.number_input("Lat", value=20.59, min_value=-90.0, max_value=90.0, format="%.6f")
            lon = c2.number_input("Lon", value=78.96, min_value=-180.0, max_value=180.0, format="%.6f")
            rad = st.number_input("Radius (meters)", value=5000, min_value=10, step=10)
            if lat and lon: 
                new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()
        elif roi_method == "Manual Coordinates":
            c1, c2 = st.columns(2)
            min_lon = c1.number_input("Min Lon", value=78.0, min_value=-180.0, max_value=180.0, format="%.6f")
            min_lat = c2.number_input("Min Lat", value=20.0, min_value=-90.0, max_value=90.0, format="%.6f")
            max_lon = c1.number_input("Max Lon", value=79.0, min_value=-180.0, max_value=180.0, format="%.6f")
            max_lat = c2.number_input("Max Lat", value=21.0, min_value=-90.0, max_value=90.0, format="%.6f")
            if min_lon < max_lon and min_lat < max_lat: 
                new_roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        if new_roi:
            if st.session_state['roi'] is None or new_roi.getInfo() != st.session_state['roi'].getInfo():
                st.session_state['roi'] = new_roi
                st.session_state['calculated'] = False
                st.toast("Target Region Locked", icon="✅")

    st.markdown("---")
    
    st.markdown("### 2. Analysis Weights (MCDA)")
    st.caption("Adjust importance factors for suitability:")
    w_rain = st.slider("Rainfall Impact (%)", 0, 100, 30)
    w_slope = st.slider("Terrain Slope (%)", 0, 100, 20)
    w_lulc = st.slider("Land Cover (%)", 0, 100, 30)
    w_soil = st.slider("Soil Type (%)", 0, 100, 20)
    
    total = w_rain + w_slope + w_lulc + w_soil
    if total != 100:
        st.warning(f"⚠️ Total weight: {total}%. Normalize to 100% for best results.")
    
    st.markdown("---")
    st.markdown("### 3. Historical Data Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", datetime.now()-timedelta(365*5)) # 5 Years default
    end = c2.date_input("To", datetime.now())

    st.markdown("###")
    if st.button("RUN HYDRO SCAN 💧"):
        if st.session_state['roi']:
            st.session_state.update({
                'calculated': True,
                'start': start.strftime("%Y-%m-%d"),
                'end': end.strftime("%Y-%m-%d"),
                'w_rain': w_rain/100.0,
                'w_slope': w_slope/100.0,
                'w_lulc': w_lulc/100.0,
                'w_soil': w_soil/100.0
            })
        else:
            st.error("❌ Error: Region of Interest (ROI) missing.")
            
    # --- ADDED: 3 Lines about the Webapp HERE in Sidebar ---
    st.markdown("""
    <div style="margin-top: 20px; color: #5c6b7f; font-size: 0.85rem; line-height: 1.4; border-top: 1px solid #d1d9e6; padding-top: 15px;">
    <strong>GeoSarovar</strong> is an advanced geospatial analytics platform designed to identify optimal rainwater harvesting sites using satellite intelligence. 
    By integrating topography, land use, soil data, and rainfall patterns, it generates precise suitability models for sustainable water management. 
    Empowering planners and communities with data-driven insights to secure water resources for the future.
    </div>
    """, unsafe_allow_html=True)

# --- 6. MAIN CONTENT ---
st.markdown("""
<div class="hud-header">
    <div>
        <div class="hud-title">GeoSarovar</div>
        <div style="color:#5c6b7f; font-size:0.9rem; margin-top:5px; font-weight:600;">ADVANCED RAINWATER HARVESTING SUITABILITY SYSTEM</div>
    </div>
    <div style="text-align:right;">
        <span class="hud-badge">LIVE SATELLITE FEED</span>
        <div style="font-family:'Rajdhani'; font-size:1.2rem; margin-top:8px; color:#00204a; font-weight:bold;">""" + datetime.now().strftime("%H:%M UTC") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:50px;">
        <h2 style="color:#00204a;">🌊 AWAITING INPUT</h2>
        <p style="color:#5c6b7f; margin-bottom:20px; font-size:1.1rem;">
            Welcome to GeoSarovar. Please configure your Area of Interest and MCDA weights in the sidebar to generate a suitability model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    m = geemap.Map(height=500, basemap="HYBRID")
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': '#00204a'}, 'Target ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    p = st.session_state
    
    col_map, col_res = st.columns([3, 1])
    
    # --- FIXED: Use default 'HYBRID' then add Esri manually to avoid KeyErrors ---
    m = geemap.Map(height=700, basemap="HYBRID") # Initial Safe Basemap
    
    # Manually add Esri World Imagery (High Res)
    esri_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m.add_tile_layer(url=esri_url, name="Esri World Imagery", attribution="Esri")
    
    m.centerObject(roi, 13)

    with st.spinner("🌧️ Processing Hydro-Geospatial Data..."):
        # 1. RAINFALL (CHIRPS)
        rain_dataset = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD") \
            .filterDate(p['start'], p['end']) \
            .select('precipitation')
        
        if rain_dataset.size().getInfo() > 0:
            rain_mean = rain_dataset.mean().clip(roi)
            min_rain, max_rain = 50, 800
            rain_norm = rain_mean.clamp(min_rain, max_rain).unitScale(min_rain, max_rain)
        else:
            st.warning("Rainfall data unavailable. Using placeholder.")
            rain_norm = ee.Image(0.5).clip(roi)
            rain_mean = rain_norm

        # 2. SLOPE (NASA DEM)
        dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
        slope = ee.Terrain.slope(dem).clip(roi)
        # 0-5 deg is best. Invert: 0->1, 30->0
        slope_norm = slope.clamp(0, 30).unitScale(0, 30)
        slope_score = ee.Image(1).subtract(slope_norm) 

        # 3. LULC (ESA WorldCover)
        lulc = ee.Image("ESA/WorldCover/v100/2020").select('Map').clip(roi)
        # High score: Bare(60), Grass(30), Shrub(20). Low score: Built(50), Water(80)
        from_list = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_list   = [0.6, 0.8, 0.8, 0.7, 0.0, 1.0, 0.0, 0.1, 0.1, 0.1]
        lulc_score = lulc.remap(from_list, to_list).rename('lulc_score')

        # 4. SOIL (OpenLandMap)
        try:
            soil_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').mean().clip(roi)
            soil_score = soil_clay.clamp(0, 50).unitScale(0, 50)
        except:
            soil_score = ee.Image(0.5).clip(roi)

        # 5. WEIGHTED OVERLAY
        suitability = (rain_norm.multiply(p['w_rain'])) \
            .add(slope_score.multiply(p['w_slope'])) \
            .add(lulc_score.multiply(p['w_lulc'])) \
            .add(soil_score.multiply(p['w_soil'])) \
            .rename('score') 

        # VISUALIZATION
        vis_params = {'min': 0, 'max': 0.8, 'palette': ['d7191c', 'fdae61', 'ffffbf', 'a6d96a', '1a9641']}
        
        m.addLayer(rain_mean, {'min': 0, 'max': 200, 'palette': ['blue', 'cyan']}, 'Rainfall (Raw)', False)
        m.addLayer(slope, {'min': 0, 'max': 30, 'palette': ['white', 'black']}, 'Slope (Raw)', False)
        m.addLayer(suitability, vis_params, 'GeoSarovar Suitability Index')

        legend_dict = {
            "Excellent Potential": "1a9641", 
            "Good Potential": "a6d96a",        
            "Moderate Potential": "ffffbf",  
            "Low Potential": "fdae61",        
            "Not Suitable": "d7191c"              
        }
        m.add_legend(title="RWH Suitability", legend_dict=legend_dict)

        # --- 6. FIX: FIND & MARK BEST SITE (Using Folium Directly) ---
        try:
            # Calculate max pixel value
            max_val = suitability.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=roi,
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            ).get('score')
            
            # Mask and find centroid
            max_pixels = suitability.eq(ee.Number(max_val))
            best_site_geom = max_pixels.reduceToVectors(
                geometry=roi, 
                scale=30, 
                geometryType='centroid', 
                labelProperty='label', 
                maxPixels=1e9,
                bestEffort=True
            )
            
            if best_site_geom.size().getInfo() > 0:
                best_point = best_site_geom.first().geometry().coordinates().getInfo()
                best_lat, best_lon = best_point[1], best_point[0]
                
                # REPLACED problematic add_marker with direct Folium Marker
                icon = folium.Icon(color='green', icon='star')
                marker = folium.Marker(
                    location=[best_lat, best_lon], 
                    popup="Best Potential Site", 
                    tooltip="Highest Suitability Score",
                    icon=icon
                )
                marker.add_to(m) # Add directly to map instance
                
                st.toast("Best Site Located!", icon="⭐")
        except Exception as e:
            st.warning(f"Note: Auto-site finder skipped ({e})")

        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">📊 AREA BREAKDOWN</div>', unsafe_allow_html=True)
            
            # Classification for stats
            suit_class = ee.Image(0).where(suitability.lt(0.2), 1) \
                .where(suitability.gte(0.2).And(suitability.lt(0.4)), 2) \
                .where(suitability.gte(0.4).And(suitability.lt(0.6)), 3) \
                .where(suitability.gte(0.6).And(suitability.lt(0.8)), 4) \
                .where(suitability.gte(0.8), 5).clip(roi)
            
            with st.spinner("Calculating..."):
                df_area = calculate_area_by_class(suit_class, roi, 30)
                if not df_area.empty:
                    name_map = {"Class 1": "Unsuitable", "Class 2": "Low", "Class 3": "Moderate", "Class 4": "Good", "Class 5": "Excellent"}
                    df_area['Class'] = df_area['Class'].map(name_map).fillna(df_area['Class'])
                    st.dataframe(df_area, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="card-label">📥 EXPORT DATA</div>', unsafe_allow_html=True)
            
            if st.button("Save to Drive (GeoTIFF)"):
                    ee.batch.Export.image.toDrive(
                    image=suitability, description=f"GeoSarovar_RWH_{datetime.now().strftime('%Y%m%d')}", 
                    scale=30, region=roi, folder='GeoSarovar_Exports'
                ).start()
                    st.toast("Export Task Started")
            
            st.markdown("---")
            map_title = st.text_input("Report Title", "GeoSarovar Site Analysis")
            if st.button("Generate Report Image"):
                    with st.spinner("Rendering High-Res Map..."):
                        buf = generate_static_map_display(
                            suitability, roi, vis_params, map_title, 
                            cmap_colors=vis_params['palette']
                        )
                        if buf:
                            st.download_button("Download JPG", buf, "GeoSarovar_Map.jpg", "image/jpeg", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with col_map:
        m.to_streamlit()

