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
import torch
import torch.nn as nn
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd

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

    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #d1d9e6;
    }

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
        margin-top: 15px;
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

# --- 3. AUTHENTICATION (GEE) ---
try:
    ee.Initialize() 
except Exception as e:
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as e2:
        st.error(f"⚠️ GEE Authentication Error: {e2}")

# --- 4. MODEL DEFINITION (CNN) ---
class SARWaterClassifier(nn.Module):
    def __init__(self):
        super(SARWaterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- STATE MANAGEMENT ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = "[CNN BASED WBE]"
if 'detected_state' not in st.session_state: st.session_state['detected_state'] = None 
if 'cnn_model_file' not in st.session_state: st.session_state['cnn_model_file'] = None

# --- 5. APP HELPER FUNCTIONS ---
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

def geojson_to_ee(geo_json):
    try:
        if geo_json['type'] == 'Polygon':
            return ee.Geometry.Polygon(geo_json['coordinates'])
        elif geo_json['type'] == 'Point':
            return ee.Geometry.Point(geo_json['coordinates'])
        return None
    except: return None

def detect_state_from_geometry(geometry):
    try:
        states = ee.FeatureCollection("FAO/GAUL/2015/level1")
        center = geometry.centroid(100)
        intersecting_state = states.filterBounds(center).first()
        state_name = intersecting_state.get('ADM1_NAME').getInfo()
        return state_name
    except: return None

# --- CNN INFERENCE HELPERS ---
@st.cache_resource
def load_cnn_model(uploaded_file):
    if uploaded_file is None: return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SARWaterClassifier()
    try:
        state_dict = torch.load(uploaded_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def predict_cnn_patches(model, img_array, device):
    """Slices image into 64x64 patches, infers, and stitches."""
    h, w, c = img_array.shape
    patch_size = 64
    stride = 64
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    img_array = np.nan_to_num(img_array)
    p2, p98 = np.percentile(img_array, (2, 98))
    img_norm = np.clip(img_array, p2, p98)
    val_min, val_max = img_norm.min(), img_norm.max()
    if val_max - val_min > 1e-6:
        img_norm = (img_norm - val_min) / (val_max - val_min)
    else:
        img_norm = np.zeros_like(img_norm)

    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_norm[y:y+patch_size, x:x+patch_size, :] 
            patch_tensor = patch.transpose(2, 0, 1)
            patches.append(patch_tensor)
            coords.append((y, x))
            
    if not patches: return full_mask

    batch_size = 32
    patch_tensor_all = torch.tensor(np.array(patches), dtype=torch.float32).to(device)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(patch_tensor_all), batch_size):
            batch = patch_tensor_all[i:i+batch_size]
            outputs = model(batch)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
            predictions.extend(preds)
            
    for (y, x), pred in zip(coords, predictions):
        if pred == 1:
            full_mask[y:y+patch_size, x:x+patch_size] = 1
            
    return full_mask

# --- STATIC MAP GENERATOR ---
def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        if isinstance(roi, ee.Geometry):
            roi_json = roi.getInfo()
            roi_bounds = roi.bounds().getInfo()['coordinates'][0]
        else:
            roi_json = roi
            roi_bounds = roi['coordinates'][0]

        lons = [p[0] for p in roi_bounds]
        lats = [p[1] for p in roi_bounds]
        min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)

        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        if height_deg == 0: height_deg = 0.001
        aspect_ratio = (width_deg * np.cos(np.radians((min_lat + max_lat) / 2))) / height_deg
        
        fig_width = 10
        fig_height = fig_width / aspect_ratio
        if fig_height > 15: fig_height = 15
        if fig_height < 4: fig_height = 4

        s2_background = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(roi).filterDate('2023-01-01', '2023-12-31')\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .median().visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])

        if 'palette' in vis_params or 'min' in vis_params:
            analysis_vis = image.visualize(**vis_params)
        else:
            analysis_vis = image

        final_image = s2_background.blend(analysis_vis)
        thumb_url = final_image.getThumbURL({'region': roi_json, 'dimensions': 800, 'format': 'png'})

        response = requests.get(thumb_url, timeout=60)
        if response.status_code != 200: return None
        img_pil = Image.open(BytesIO(response.content))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
        ax.imshow(img_pil, extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        if is_categorical and class_names and 'palette' in vis_params:
            patches = [mpatches.Patch(color=c, label=n) for n, c in zip(class_names, vis_params['palette'])]
            ax.legend(handles=patches, loc='lower center', ncol=len(class_names))

        buf = BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

# --- 6. SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/nitesh4004/GeoSarovar/main/geosarovar.png", use_container_width=True)
    st.markdown("### 1. Select Module")
    # UPDATED: Removed RWH, Rainfall, Water Quality
    app_mode = st.radio("Choose Functionality:",
                        ["[CNN BASED WBE]",
                         "⚠️ Encroachment (S1 SAR)",
                         "Flood Extent Mapping"],
                        label_visibility="collapsed")
    st.markdown("---")

    # --- LOCATION LOGIC ---
    st.markdown("### 2. Location (ROI)")
    roi_method = st.radio("Selection Mode", ["Upload KML", "Point & Buffer", "Draw on Map"], label_visibility="collapsed")
    new_roi = None
    
    if roi_method == "Upload KML":
        kml = st.file_uploader("Upload KML", type=['kml'])
        if kml: 
            new_roi = parse_kml(kml.read())
            if new_roi: st.session_state['roi'] = new_roi.simplify(maxError=50)

    elif roi_method == "Point & Buffer":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", value=20.59)
        lon = c2.number_input("Lon", value=78.96)
        rad = st.number_input("Radius (m)", value=5000)
        new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()
        if new_roi: st.session_state['roi'] = new_roi
            
    elif roi_method == "Draw on Map":
        if st.session_state['roi'] is not None:
             st.info(f"📍 ROI Set: {st.session_state.get('detected_state', 'Custom Area')}")
             if st.button("🗑️ Reset / Draw New"):
                st.session_state['roi'] = None
                st.session_state['calculated'] = False
                st.session_state['detected_state'] = None
                st.rerun()

    if roi_method != "Draw on Map" and st.session_state['roi']:
        if not st.session_state['detected_state']:
            with st.spinner("Detecting Location..."):
                detected = detect_state_from_geometry(st.session_state['roi'])
                if detected:
                    st.session_state['detected_state'] = detected
                    st.success(f"ROI Locked ✅ ({detected})")
                else:
                    st.success("ROI Locked ✅")
    
    st.markdown("---")

    # --- PARAMETERS ---
    params = {}
    
    if app_mode == "[CNN BASED WBE]":
        st.markdown("### 3. Deep Learning Config")
        model_file = st.file_uploader("Upload Model (.pt)", type=['pt'], help="Upload your 'sar_water_classifier.pt'")
        if model_file:
            st.session_state['cnn_model_file'] = model_file
        st.markdown("**Sentinel-1 Data Date**")
        target_date = st.date_input("Target Date", datetime(2023, 8, 15))
        params = {
            'target_date': target_date.strftime("%Y-%m-%d"),
            'model_uploaded': st.session_state.get('cnn_model_file') is not None
        }
        st.warning("⚠️ **Note:** Analysis runs on CPU. Choose a small ROI (e.g., 5km radius).")

    elif app_mode == "⚠️ Encroachment (S1 SAR)":
        st.markdown("### 3. Comparison Dates")
        orbit = st.radio("Orbit Pass", ["BOTH", "ASCENDING", "DESCENDING"])
        col1, col2 = st.columns(2)
        d1_start = col1.date_input("Start 1", datetime(2018, 6, 1))
        d1_end = col2.date_input("End 1", datetime(2018, 9, 30))
        col3, col4 = st.columns(2)
        d2_start = col3.date_input("Start 2", datetime(2024, 6, 1))
        d2_end = col4.date_input("End 2", datetime(2024, 9, 30))
        params = {'d1_start': d1_start.strftime("%Y-%m-%d"), 'd1_end': d1_end.strftime("%Y-%m-%d"), 'd2_start': d2_start.strftime("%Y-%m-%d"), 'd2_end': d2_end.strftime("%Y-%m-%d"), 'orbit': orbit}

    elif app_mode == "Flood Extent Mapping":
        st.markdown("### 3. Flood Event Details")
        orbit = st.radio("Orbit Pass", ["BOTH", "ASCENDING", "DESCENDING"])
        col1, col2 = st.columns(2)
        pre_start = col1.date_input("Pre Start", datetime(2023, 4, 1))
        pre_end = col2.date_input("Pre End", datetime(2023, 6, 1))
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
            st.error("Please draw or select an ROI first.")

# --- 7. MAIN CONTENT ---
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

# Helper for Safe Map Loading
def get_safe_map(height=500):
    m = geemap.Map(height=height, basemap="HYBRID")
    return m

# --- CASE 1: DRAW MODE ACTIVE, ROI NOT SET ---
if roi_method == "Draw on Map" and st.session_state['roi'] is None:
    st.info("🗺️ **Instructions:**\n1. Use the **Polygon** or **Rectangle** tool on the map sidebar.\n2. Draw your area of interest.\n3. Click the **'✅ Set as ROI'** button below to lock it.")
    m_draw = geemap.Map(height=550, basemap="HYBRID", center=[20.59, 78.96], zoom=5)
    map_output = m_draw.to_streamlit(width=None, height=550)
    
    if st.button("✅ Set as ROI", type="primary"):
        if map_output and isinstance(map_output, dict) and map_output.get('last_active_drawing'):
            drawn_geom = map_output['last_active_drawing']['geometry']
            ee_geom = geojson_to_ee(drawn_geom)
            if ee_geom:
                st.session_state['roi'] = ee_geom
                with st.spinner("Locking Region & Detecting State..."):
                    detected = detect_state_from_geometry(ee_geom)
                    st.session_state['detected_state'] = detected if detected else "Custom Area"
                st.success("ROI Locked! Please click 'RUN ANALYSIS' in the sidebar.")
                st.rerun()
        else:
            st.warning("⚠️ No drawing detected! Please draw a polygon on the map first.")

# --- CASE 2: ROI IS SET BUT NOT CALCULATED YET ---
elif not st.session_state['calculated']:
    st.info(f"👈 ROI Locked ({st.session_state.get('detected_state', 'Unknown')}). Please click **RUN ANALYSIS** in the sidebar.")
    m = get_safe_map(500)
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': 'yellow'}, 'ROI')
    m.to_streamlit(height=500)

# --- CASE 3: ANALYSIS RESULTS ---
else:
    roi = st.session_state['roi']
    mode = st.session_state['mode']
    p = st.session_state['params']

    col_map, col_res = st.columns([3, 1])
    m = get_safe_map(700)
    m.centerObject(roi, 13)
    image_to_export = None
    vis_export = {}

    # ==========================================
    # LOGIC: CNN BASED WATER BODY EXTRACTION
    # ==========================================
    if mode == "[CNN BASED WBE]":
        if not p['model_uploaded']:
            st.error("❌ Please upload the 'sar_water_classifier.pt' model file in the sidebar first!")
        else:
            with st.spinner("Initializing Hybrid Engine (GEE + PyTorch)..."):
                model, device = load_cnn_model(st.session_state['cnn_model_file'])
                if model:
                    st.write("Fetching S1 Radar Data...")
                    s1 = ee.ImageCollection("COPERNICUS/S1_GRD")\
                        .filterBounds(roi)\
                        .filterDate(p['target_date'], datetime.strptime(p['target_date'], "%Y-%m-%d") + timedelta(days=15))\
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                        .select(['VV', 'VH'])
                    
                    if s1.size().getInfo() > 0:
                        img_ee = s1.mean().clip(roi)
                        try:
                            st.write("Downloading image chips to local memory...")
                            arr = geemap.ee_to_numpy(img_ee, region=roi, scale=20)
                            
                            if arr is not None:
                                st.write(f"Running CNN Inference on {device}...")
                                mask = predict_cnn_patches(model, arr, device)
                                st.success("Extraction Complete!")
                                
                                with col_map:
                                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                                    vv = arr[:,:,0]
                                    vv_disp = (vv - np.min(vv)) / (np.max(vv) - np.min(vv))
                                    ax[0].imshow(vv_disp, cmap='gray')
                                    ax[0].set_title("Sentinel-1 Input (VV)", fontweight='bold')
                                    ax[0].axis('off')
                                    
                                    cmap_cust = mcolors.ListedColormap(['black', 'cyan'])
                                    ax[1].imshow(mask, cmap=cmap_cust)
                                    ax[1].set_title("CNN Water Extraction", fontweight='bold')
                                    ax[1].axis('off')
                                    water_patch = mpatches.Patch(color='cyan', label='Water Body')
                                    ax[1].legend(handles=[water_patch], loc='lower right')
                                    st.pyplot(fig)
                                    st.info("Note: Result is displayed as a static plot because it was processed locally by the CNN.")

                                with col_res:
                                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-label">🧠 CNN STATS</div>', unsafe_allow_html=True)
                                    pixel_area = 20 * 20 
                                    water_pixels = np.sum(mask)
                                    area_ha = (water_pixels * pixel_area) / 10000
                                    st.metric("Water Area", f"{area_ha:.2f} Ha")
                                    st.caption(f"Based on {p['target_date']}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                                    buf = BytesIO()
                                    mask_img.save(buf, format="PNG")
                                    st.download_button("Download Mask (PNG)", buf.getvalue(), "cnn_water_mask.png", "image/png")
                            else:
                                st.error("Failed to download image data (Region might be too large or empty).")
                        except Exception as e:
                            st.error(f"Download/Inference Failed: {e}")
                    else:
                        st.warning(f"No Sentinel-1 data found near {p['target_date']}.")

    # ==========================================
    # LOGIC C: ENCROACHMENT DETECTION
    # ==========================================
    elif mode == "⚠️ Encroachment (S1 SAR)":
        with st.spinner("Processing Sentinel-1 SAR Data..."):
            def get_sar_collection(start_d, end_d, roi_geom, orbit_pass):
                s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(start_d, end_d).filterBounds(roi_geom)
                if orbit_pass != "BOTH": s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
                return s1

            def process_water_mask(col, roi_geom):
                if col.size().getInfo() == 0: return None, "N/A"
                date_found = ee.Date(col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                def speckle_filter(img): return img.select('VV').focal_median(50, 'circle', 'meters').rename('VV_smoothed')
                mosaic = col.map(speckle_filter).min().clip(roi_geom)
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
                    vis_export = {'min': 1, 'max': 3, 'palette': ['cyan', 'red', 'blue']}

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
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Insufficient SAR data.")
            except Exception as e:
                st.error(f"Computation Error: {e}")

    # ==========================================
    # LOGIC D: FLOOD EXTENT MAPPING
    # ==========================================
    elif mode == "Flood Extent Mapping":
        with st.spinner("Processing Flood Extent..."):
            try:
                collection = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('resolution_meters', 10)).filterBounds(roi).select('VH')
                if p['orbit'] != "BOTH": collection = collection.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))

                before_col = collection.filterDate(p['pre_start'], p['pre_end'])
                after_col = collection.filterDate(p['post_start'], p['post_end'])

                if before_col.size().getInfo() > 0 and after_col.size().getInfo() > 0:
                    date_pre = ee.Date(before_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    date_post = ee.Date(after_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    before = before_col.median().clip(roi)
                    after = after_col.mosaic().clip(roi)
                    before_f = before.focal_mean(50, 'circle', 'meters')
                    after_f = after.focal_mean(50, 'circle', 'meters')

                    difference = after_f.divide(before_f)
                    difference_binary = difference.gt(p['threshold'])
                    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                    permanent_water_mask = gsw.select('occurrence').gt(30)
                    flooded = difference_binary.updateMask(permanent_water_mask.Not())
                    dem = ee.Image('WWF/HydroSHEDS/03VFDEM')
                    slope = ee.Algorithms.Terrain(dem).select('slope')
                    flooded = flooded.updateMask(slope.lt(5)).updateMask(flooded.connectedPixelCount().gte(8)).selfMask()

                    image_to_export = flooded
                    vis_export = {'min': 0, 'max': 1, 'palette': ['#0000FF']}

                    m.addLayer(before_f, {'min': -25, 'max': 0}, 'Before Flood (Dry)', False)
                    m.addLayer(after_f, {'min': -25, 'max': 0}, 'After Flood (Wet)', True)
                    m.addLayer(flooded, {'palette': ['#0000FF']}, '🌊 Estimated Flood Extent')

                    flood_stats = flooded.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, bestEffort=True)
                    flood_area_ha = round(flood_stats.values().get(0).getInfo() / 10000, 2)

                    with col_res:
                        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                        st.markdown("### 🌊 Flood Report")
                        st.metric("Estimated Extent", f"{flood_area_ha} Ha")
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"No images found for Orbit: {p['orbit']} in these dates.")

            except Exception as e:
                st.error(f"Error: {e}")

    # --- EXPORT TOOLS ---
    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📥 EXPORTS</div>', unsafe_allow_html=True)

        if st.button("Save to Drive (GeoTIFF)"):
            if image_to_export:
                desc = f"GeoSarovar_{mode.split(' ')[0]}_{datetime.now().strftime('%Y%m%d')}"
                ee.batch.Export.image.toDrive(image=image_to_export, description=desc, scale=30, region=roi, folder='GeoSarovar_Exports').start()
                st.toast("Export started! Check Google Drive.")
            else:
                st.warning("No result to export.")

        st.markdown("---")
        report_title = st.text_input("Report Title", f"Analysis: {mode}")
        if st.button("Generate Map Image"):
            with st.spinner("Rendering..."):
                if image_to_export:
                    is_cat = False
                    c_names = None
                    cmap = None
                    if mode == "Flood Extent Mapping": is_cat = True; c_names = ['Flood Extent']
                    elif mode == "⚠️ Encroachment (S1 SAR)": is_cat = True; c_names = ['Stable Water', 'Encroachment', 'New Water']
                    elif 'palette' in vis_export: cmap = vis_export['palette']
                    buf = generate_static_map_display(image_to_export, roi, vis_export, report_title, cmap_colors=cmap, is_categorical=is_cat, class_names=c_names)
                    if buf: st.download_button("Download JPG", buf, "GeoSarovar_Map.jpg", "image/jpeg", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if mode != "[CNN BASED WBE]": 
        with col_map:
            m.to_streamlit(height=700)
