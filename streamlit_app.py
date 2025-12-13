import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import os
import tempfile
import requests
import geopandas as gpd
import zipfile
import gdown
from datetime import datetime, timedelta
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="GeoSarovar - SAR Water Intelligence", 
    page_icon="📡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;600&display=swap');
    :root { --accent-primary: #00204a; --text-primary: #00204a; }
    .stApp { background-color: #ffffff; font-family: 'Inter', sans-serif; color: var(--text-primary); }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; color: var(--accent-primary) !important; }
    .hud-header { background: #fff; border-bottom: 2px solid #00204a; padding: 15px 25px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .hud-title { font-family: 'Rajdhani'; font-size: 2rem; font-weight: 700; color: #00204a; }
    .glass-card { background: #fff; border: 1px solid #e0e0e0; padding: 20px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .card-label { font-family: 'Rajdhani'; font-weight: 700; font-size: 1.1rem; border-bottom: 2px solid #f0f0f0; margin-bottom: 10px; color: #00204a; }
    div.stButton > button { background: #00204a; color: white; border-radius: 6px; font-family: 'Rajdhani'; font-weight: 700; width: 100%; }
    div.stButton > button:hover { background: #005792; }
    </style>
""", unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
try:
    service_account = st.secrets["gcp_service_account"]["client_email"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
    ee.Initialize(credentials)
except:
    try: ee.Initialize()
    except Exception as e: st.error(f"Authentication Error: {e}"); st.stop()

# --- STATE ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None

# --- 4. HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def load_admin_data(url, is_gdrive=False):
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")
        if is_gdrive: gdown.download(url, zip_path, quiet=True, fuzzy=True)
        else:
            r = requests.get(url)
            with open(zip_path, "wb") as f: f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.endswith(".shp") or f.endswith(".geojson"):
                    gdf = gpd.read_file(os.path.join(root, f))
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    col_map = {'STATE_UT': 'STATE', 'State': 'STATE', 'Name': 'District', 'Sub_dist': 'Subdistrict'}
                    gdf.rename(columns=col_map, inplace=True)
                    return gdf
    except: return None

def geopandas_to_ee(gdf_row):
    try:
        gjson = json.loads(gdf_row.geometry.to_json())
        return ee.Geometry(gjson['features'][0]['geometry'] if 'features' in gjson else gjson)
    except: return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/nitesh4004/GeoSarovar/main/geosarovar.png", use_container_width=True)
    st.markdown("### 1. Select Module")
    app_mode = st.radio("Choose Mode:", ["📍 RWH Site Suitability", "📡 SAR Encroachment (Sentinel-1)"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 2. Location (ROI)")
    roi_type = st.radio("Mode", ["Select Admin Boundary", "Point & Buffer"], label_visibility="collapsed")
    
    new_roi = None
    if roi_type == "Select Admin Boundary":
        lvl = st.selectbox("Level", ["Districts", "Subdistricts", "States"])
        url = 'https://drive.google.com/uc?id=1tMyiUheQBcwwPwZQla67PwC5-AqenTmv' if lvl == "Districts" else ('https://drive.google.com/uc?id=18lMyt2j3Xjz_Qk_2Kzppr8EVlVDx_yOv' if lvl == "Subdistricts" else "https://github.com/nitesh4004/GeoFormatX/raw/main/STATE_BOUNDARY.zip")
        is_drive = lvl in ["Districts", "Subdistricts"]
        
        with st.spinner("Loading Map..."):
            gdf = load_admin_data(url, is_drive)
        if gdf is not None:
            if 'STATE' in gdf.columns:
                st_sel = st.selectbox("State", sorted(gdf['STATE'].astype(str).unique()))
                gdf = gdf[gdf['STATE'] == st_sel]
                if 'District' in gdf.columns and not gdf.empty:
                    d_sel = st.selectbox("District", sorted(gdf['District'].astype(str).unique()))
                    gdf = gdf[gdf['District'] == d_sel]
                    if 'Subdistrict' in gdf.columns and not gdf.empty:
                        s_sel = st.selectbox("Subdistrict", sorted(gdf['Subdistrict'].astype(str).unique()))
                        gdf = gdf[gdf['Subdistrict'] == s_sel]
            if not gdf.empty:
                new_roi = geopandas_to_ee(gdf.iloc[[0]])
                st.info(f"Selected: {len(gdf)} Feature")

    elif roi_type == "Point & Buffer":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", 20.59)
        lon = c2.number_input("Lon", 78.96)
        rad = st.number_input("Radius (m)", 5000)
        new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()

    if new_roi:
        st.session_state['roi'] = new_roi.simplify(50)
        st.success("Location Locked ✅")

    st.markdown("---")
    params = {}
    
    if app_mode == "📍 RWH Site Suitability":
        w_rain = st.slider("Rainfall %", 0, 100, 30)
        w_slope = st.slider("Slope %", 0, 100, 20)
        w_lulc = st.slider("Land Cover %", 0, 100, 30)
        w_soil = st.slider("Soil %", 0, 100, 20)
        start = st.date_input("Start", datetime.now()-timedelta(1825))
        end = st.date_input("End", datetime.now())
        params = {'w_rain': w_rain/100, 'w_slope': w_slope/100, 'w_lulc': w_lulc/100, 'w_soil': w_soil/100, 'start': start, 'end': end}

    elif app_mode == "📡 SAR Encroachment (Sentinel-1)":
        st.markdown("### 3. Comparison Dates")
        st.info("Select two specific months to compare water extent.")
        
        col1, col2 = st.columns(2)
        # Initial Date
        init_date = col1.date_input("Initial Date", datetime(2020, 10, 1))
        # Final Date
        final_date = col2.date_input("Final Date", datetime(2024, 10, 1))
        
        st.markdown("### 4. SAR Settings")
        threshold = st.slider("Water Threshold (dB)", -30, -10, -18, help="Lower value = stricter water detection. Water is usually <-18dB in VH.")
        
        params = {'init': init_date, 'final': final_date, 'thresh': threshold}

    if st.button("RUN ANALYSIS 🚀"):
        if st.session_state['roi']:
            st.session_state['calculated'] = True
            st.session_state['mode'] = app_mode
            st.session_state['params'] = params
        else: st.error("Select Location first.")

# --- 6. MAIN CONTENT ---
st.markdown(f"""
<div class="hud-header">
    <div class="hud-title">GeoSarovar <span style='font-size:1rem; font-weight:400; color:#5c6b7f;'>| {app_mode}</span></div>
    <div style="background:#e6f0ff; color:#00204a; padding:5px 15px; border-radius:20px; font-weight:bold;">LIVE SYSTEM</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.info("👈 Configure inputs in the sidebar.")
    m = geemap.Map(height=500, basemap="HYBRID")
    if st.session_state['roi']: m.centerObject(st.session_state['roi'], 12); m.addLayer(st.session_state['roi'], {'color':'yellow'}, 'ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    mode = st.session_state['mode']
    p = st.session_state['params']
    
    # === SAR ENCROACHMENT LOGIC ===
    if mode == "📡 SAR Encroachment (Sentinel-1)":
        c_map, c_stats = st.columns([3, 1])
        
        with st.spinner("Processing Sentinel-1 Radar Data..."):
            # 1. Load S1 Collection
            def get_s1_image(date):
                # Filter 15 days around the selected date
                start = ee.Date(date.strftime("%Y-%m-%d")).advance(-15, 'day')
                end = ee.Date(date.strftime("%Y-%m-%d")).advance(15, 'day')
                
                s1 = ee.ImageCollection("COPERNICUS/S1_GRD")\
                    .filterBounds(roi)\
                    .filterDate(start, end)\
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                    .select('VH')
                
                # Smoothening to remove Speckle Noise
                return s1.median().focal_median(50, 'circle', 'meters').clip(roi)

            img_init = get_s1_image(p['init'])
            img_final = get_s1_image(p['final'])

            # 2. Water Masking (Thresholding)
            # Water < Threshold (e.g. -18)
            water_init = img_init.lt(p['thresh']).selfMask()
            water_final = img_final.lt(p['thresh']).selfMask()

            # 3. Change Detection
            # Encroachment = Was Water (1) AND Is Now Land (0)
            # We unmask to handle nulls as 0
            w1 = water_init.unmask(0)
            w2 = water_final.unmask(0)
            
            # Encroachment: 1 -> 0
            encroachment = w1.And(w2.Not()).selfMask()
            # New Water: 0 -> 1
            new_water = w1.Not().And(w2).selfMask()

            # 4. Map Visualization
            m = geemap.Map(height=650, basemap="HYBRID")
            m.centerObject(roi, 13)
            
            # Layer 1: Initial Water
            m.addLayer(water_init, {'palette': ['blue']}, f"Initial Water ({p['init']})")
            
            # Layer 2: Final Water
            m.addLayer(water_final, {'palette': ['cyan']}, f"Final Water ({p['final']})")
            
            # Layer 3: Change (Red = Loss, Green = Gain)
            m.addLayer(encroachment, {'palette': ['ff0000']}, '⚠️ Encroachment (Water Loss)')
            m.addLayer(new_water, {'palette': ['00ff00']}, 'New Water Body')

            # --- SPLIT MAP (SWIPE) ---
            # Create two images for swipe: Composite of Radar + Water Mask
            def make_vis(img, mask, color):
                # Background Radar (Gray) + Water (Color)
                radar_vis = img.visualize(min=-30, max=0)
                mask_vis = mask.visualize(palette=[color])
                return radar_vis.blend(mask_vis)

            left_layer = make_vis(img_init, water_init, 'blue')
            right_layer = make_vis(img_final, water_final, 'cyan')
            
            m.split_map(left_layer, right_layer)

            # --- TIME LAPSE GIF GENERATION ---
            # Generate monthly composite for the duration
            st.markdown("### ⏳ Temporal Analysis")
            with st.spinner("Generating Time-Lapse..."):
                try:
                    # Define full timeline
                    tl_col = ee.ImageCollection("COPERNICUS/S1_GRD")\
                        .filterBounds(roi)\
                        .filterDate(str(p['init']), str(p['final']))\
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .select('VH')
                    
                    # Create monthly composites to smooth animation
                    def create_monthly(m_offset):
                        d = ee.Date(str(p['init'])).advance(m_offset, 'month')
                        img = tl_col.filterDate(d, d.advance(1, 'month')).median().clip(roi)
                        # Water mask visualization
                        mask = img.lt(p['thresh'])
                        return mask.visualize(palette=['black', 'blue']).set({'system:time_start': d.millis()})

                    # Number of months
                    n_months = (p['final'].year - p['init'].year) * 12 + (p['final'].month - p['init'].month)
                    if n_months > 0:
                        seq = ee.List.sequence(0, n_months)
                        tl_images = ee.ImageCollection(seq.map(create_monthly))
                        
                        # Export GIF
                        vid_args = {
                            'dimensions': 600,
                            'region': roi,
                            'framesPerSecond': 2,
                            'crs': 'EPSG:3857'
                        }
                        gif_url = tl_images.getVideoThumbURL(vid_args)
                        st.image(gif_url, caption="Water Body Time-Lapse (Blue=Water)", use_container_width=True)
                    else:
                        st.warning("Date range too short for Time-Lapse.")
                except Exception as e:
                    st.warning(f"Time-lapse generation skipped: {e}")

        # --- STATISTICS CARD ---
        with c_stats:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">📊 CHANGE STATS</div>', unsafe_allow_html=True)
            
            # Calculate Areas
            def get_area(img):
                area = img.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e9, bestEffort=True
                )
                return area.get('VH').getInfo() if area.get('VH') else 0

            enc_sqm = get_area(encroachment)
            new_sqm = get_area(new_water)
            
            enc_ha = round(enc_sqm / 10000, 2)
            new_ha = round(new_sqm / 10000, 2)
            net_change = round(new_ha - enc_ha, 2)

            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="font-size:0.9rem; color:#777;">Encroached Area (Lost):</div>
                <div style="font-size:1.5rem; font-weight:bold; color:#d32f2f;">{enc_ha} Ha 📉</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-size:0.9rem; color:#777;">New Water Area (Gained):</div>
                <div style="font-size:1.5rem; font-weight:bold; color:#388e3c;">{new_ha} Ha 📈</div>
            </div>
            <hr>
            <div>
                <div style="font-size:0.9rem; color:#777;">Net Water Change:</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#00204a;">{net_change} Ha</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Legend
            st.markdown('<div class="glass-card"><b>Layer Legend</b><br>🟦 Initial Water<br>🟦 Final Water<br>🟥 Encroachment</div>', unsafe_allow_html=True)

        with c_map:
            m.to_streamlit()

    # === ORIGINAL RWH LOGIC (PRESERVED) ===
    elif mode == "📍 RWH Site Suitability":
        c_map, c_stats = st.columns([3, 1])
        with st.spinner("Analyzing Terrain & Hydrology..."):
            rain = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(str(p['start']), str(p['end'])).select('precipitation').mean().clip(roi).clamp(50,800).unitScale(50,800)
            slope = ee.Terrain.slope(ee.Image("NASA/NASADEM_HGT/001").select('elevation')).clip(roi).clamp(0,30).unitScale(0,30).multiply(-1).add(1)
            lulc = ee.Image("ESA/WorldCover/v100/2020").select('Map').clip(roi).remap([10,20,30,40,50,60,80], [0.6,0.8,0.8,0.7,0.0,1.0,0.0])
            try: soil = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').mean().clip(roi).clamp(0,50).unitScale(0,50)
            except: soil = ee.Image(0.5).clip(roi)
            
            suitability = (rain.multiply(p['w_rain'])).add(slope.multiply(p['w_slope'])).add(lulc.multiply(p['w_lulc'])).add(soil.multiply(p['w_soil']))
            
            m = geemap.Map(height=700, basemap="HYBRID")
            m.centerObject(roi, 13)
            vis = {'min': 0, 'max': 0.8, 'palette': ['d7191c', 'fdae61', 'ffffbf', 'a6d96a', '1a9641']}
            m.addLayer(suitability, vis, 'Suitability')
            
            with c_stats:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**RWH Potential Score**")
                # Simple Histogram
                hist = suitability.reduceRegion(ee.Reducer.histogram(), roi, 100).get('constant').getInfo()
                if hist: st.bar_chart(pd.DataFrame(hist['histogram'], index=[round(x,1) for x in hist['bucketMeans']]))
                st.markdown("</div>", unsafe_allow_html=True)
                
        with c_map:
            m.to_streamlit()
