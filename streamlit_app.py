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
    # Don't stop, as DL module might work without GEE if using Planetary Computer

# --- STATE MANAGEMENT ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = "📍 RWH Site Suitability"
if 'dl_result' not in st.session_state: st.session_state['dl_result'] = None

# --- 4. DL MODEL HELPERS (From Inference Utils) ---

def build_model():
    """Construct the segmentation model architecture (U-Net ResNet-34)."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    )
    return model

@st.cache_resource
def load_dl_model_from_drive(device="cpu"):
    """Download and load the model from GDrive."""
    model_path = "water_unet_best.pth"
    file_id = "1-v-SLRDr3OiiKAnQeebpwQzIPDpLamsW"
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading Model weights from Server (One time)..."):
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
    tile = tile.astype(np.float32)
    tile = tile / 255.0
    return tile

def predict_large_image(model, image, device="cpu", tile_size=512, overlap=64):
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
    
    if len(geoms) == 0:
        return gpd.GeoDataFrame({"id": [], "area_km2": [], "geometry": []}, crs=crs)

    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    gdf["id"] = range(1, len(gdf) + 1)
    
    if crs and not isinstance(crs, str) and crs.is_geographic:
        gdf_proj = gdf.to_crs(epsg=3857)
        gdf["area_m2"] = gdf_proj.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area
        
    gdf["area_km2"] = (gdf["area_m2"] / 1_000_000).round(4)
    gdf = gdf.sort_values("area_km2", ascending=False).reset_index(drop=True)
    return gdf

def build_planetary_computer_image_for_aoi(aoi_geojson, satellite_type: str, months_back: int = 6):
    import pystac_client
    import planetary_computer
    import stackstac
    
    # Handle GeoJSON input logic
    if isinstance(aoi_geojson, dict) and "geometry" in aoi_geojson:
        geom_dict = aoi_geojson["geometry"]
    else:
        geom_dict = aoi_geojson
        
    coords = geom_dict["coordinates"][0] if geom_dict['type'] == 'Polygon' else geom_dict["coordinates"][0][0]
    # Simple bbox extract
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox_wgs84 = [min(lons), min(lats), max(lons), max(lats)]

    # Web Mercator Transform
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    min_x, min_y = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    max_x, max_y = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    bbox_mercator = [min_x, min_y, max_x, max_y]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    if "Sentinel-2" in satellite_type:
        collection = "sentinel-2-l2a"
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
        scale = 10
    else: # Landsat
        collection = "landsat-c2-l2"
        bands = ["coastal", "blue", "green", "red", "nir08", "swir16"]
        scale = 30

    search = catalog.search(
        collections=[collection], bbox=bbox_wgs84, datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.items())
    image_count = len(items)
    if image_count == 0: return None, None, scale, 0

    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[:10]
    
    stack = stackstac.stack(
        items_sorted, assets=bands, bounds=bbox_mercator, epsg=3857, resolution=scale,
    )
    
    import dask
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        composite = stack.median(dim="time").compute()

    image = composite.values
    # Transform creation
    x_coords = composite.x.values
    y_coords = composite.y.values
    x_res = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else scale
    y_res = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -scale
    transform = Affine(x_res, 0, float(x_coords[0]), 0, y_res, float(y_coords[0]))

    image = np.nan_to_num(image, nan=0.0)
    image = np.clip(image, 0, 65535).astype(np.uint16)
    
    profile = {
        'driver': 'GTiff', 'height': image.shape[1], 'width': image.shape[2],
        'count': image.shape[0], 'dtype': 'uint16', 'crs': 'EPSG:3857', 'transform': transform,
    }
    
    from rasterio.coords import BoundingBox
    bounds_tuple = rasterio.transform.array_bounds(image.shape[1], image.shape[2], transform)
    bounds = BoundingBox(*bounds_tuple)

    return image, profile, transform, 'EPSG:3857', bounds, image_count

def read_geotiff(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return image, profile, transform, crs, bounds

# --- 5. APP HELPER FUNCTIONS (Existing) ---

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

# --- ADVANCED STATIC MAP GENERATOR ---
def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        if isinstance(roi, ee.Geometry):
            try:
                roi_json = roi.getInfo() 
                roi_bounds = roi.bounds().getInfo()['coordinates'][0]
            except: return None
        else:
            roi_json = roi
            roi_bounds = roi['coordinates'][0]

        lons = [p[0] for p in roi_bounds]
        lats = [p[1] for p in roi_bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        if height_deg == 0: height_deg = 0.001
        
        aspect_ratio = (width_deg * np.cos(np.radians((min_lat + max_lat) / 2))) / height_deg
        fig_width = 12 
        fig_height = fig_width / aspect_ratio
        if fig_height > 20: fig_height = 20
        if fig_height < 4: fig_height = 4

        # Background Imagery (Sentinel-2 Cloud Free)
        s2_background = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(roi).filterDate('2023-01-01', '2023-12-31')\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .median().visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
        
        if 'palette' in vis_params or 'min' in vis_params:
            analysis_vis = image.visualize(**vis_params)
        else:
            analysis_vis = image 
            
        final_image = s2_background.blend(analysis_vis)

        thumb_url = final_image.getThumbURL({
            'region': roi_json, 'dimensions': 1000, 'format': 'png', 'crs': 'EPSG:4326'
        })
        
        response = requests.get(thumb_url, timeout=120)
        if response.status_code != 200: return None
            
        img_pil = Image.open(BytesIO(response.content))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#ffffff')
        extent = [min_lon, max_lon, min_lat, max_lat]
        ax.imshow(img_pil, extent=extent, aspect='auto')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#00204a')
        
        # Grid and Ticks
        ax.tick_params(colors='black', labelsize=10)
        for spine in ax.spines.values(): spine.set_edgecolor('black')
        
        # Legend Logic
        if is_categorical and class_names and 'palette' in vis_params:
            patches = [mpatches.Patch(color=c, label=n) for n, c in zip(class_names, vis_params['palette'])]
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                             frameon=False, ncol=min(len(class_names), 4))
        elif cmap_colors and 'min' in vis_params:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
            norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Index Value', color='black', fontsize=12)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

# --- 6. SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/nitesh4004/GeoSarovar/main/geosarovar.png", use_container_width=True)
    st.markdown("### 1. Select Module")
    app_mode = st.radio("Choose Functionality:", 
                        ["📍 RWH Site Suitability", 
                         "⚠️ Encroachment (S1 SAR)", 
                         "Flood Extent Mapping", 
                         "🧪 Water Quality",
                         "🤖 DL Water Segmentation"], 
                        label_visibility="collapsed")
    st.markdown("---")
    
    # Specific Logic for DL Module inputs
    if app_mode == "🤖 DL Water Segmentation":
        st.markdown("### 2. DL Input Source")
        dl_source = st.radio("Input Type", ["Use ROI (Planetary Computer)", "Upload GeoTIFF"], label_visibility="collapsed")
        
        params = {}
        if dl_source == "Use ROI (Planetary Computer)":
            st.markdown("### 3. Location (ROI)")
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
            
            sat_type = st.selectbox("Satellite", ["Sentinel-2", "Landsat 8", "Landsat 9"])
            params = {'source': 'pc', 'sat_type': sat_type}
            
        else: # Upload GeoTIFF
            uploaded_file = st.file_uploader("Upload 6-Band GeoTIFF", type=["tif", "tiff"])
            params = {'source': 'upload', 'file': uploaded_file}

    # Standard GEE Input Logic for other modules
    else: 
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
            st.markdown("### 3. Criteria Weights")
            st.info("MCDA - Multi-Criteria Decision Analysis")
            w_rain = st.slider("Precipitation Potential", 0, 100, 20)
            w_slope = st.slider("Terrain & Drainage (TWI)", 0, 100, 40, help="Combines Slope and Topographic Wetness")
            w_lulc = st.slider("Land Use Availability", 0, 100, 20)
            w_soil = st.slider("Soil Retention (Clay)", 0, 100, 20)
            
            total_w = w_rain + w_slope + w_lulc + w_soil
            st.markdown(f"**Total Weight: {total_w}** (Normalized automatically)")

            st.markdown("### 4. Period")
            start = st.date_input("From", datetime.now()-timedelta(365*5))
            end = st.date_input("To", datetime.now())
            params = {
                'w_rain': w_rain, 'w_slope': w_slope, 'w_lulc': w_lulc, 'w_soil': w_soil, 
                'start': start.strftime("%Y-%m-%d"), 'end': end.strftime("%Y-%m-%d"), 'total': total_w
            }

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
        
        elif app_mode == "Flood Extent Mapping":
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

        elif app_mode == "🧪 Water Quality":
            st.markdown("### 3. Monitoring Config")
            wq_param = st.selectbox("Parameter", ["Turbidity (NDTI)", "Total Suspended Solids (TSS)", "Cyanobacteria Index", "Chlorophyll-a", "CDOM (Organic Matter)"])
            st.markdown("**Timeframe**")
            col1, col2 = st.columns(2)
            wq_start = col1.date_input("Start", datetime.now()-timedelta(days=90))
            wq_end = col2.date_input("End", datetime.now())
            cloud_thresh = st.slider("Max Cloud Cover %", 5, 50, 20)
            params = {'param': wq_param, 'start': wq_start.strftime("%Y-%m-%d"), 'end': wq_end.strftime("%Y-%m-%d"), 'cloud': cloud_thresh}

    st.markdown("###")
    if st.button("RUN ANALYSIS 🚀"):
        if app_mode == "🤖 DL Water Segmentation" and params.get('source') == 'upload' and params.get('file') is None:
            st.error("Please upload a file.")
        elif st.session_state['roi'] or (app_mode == "🤖 DL Water Segmentation" and params.get('source') == 'upload'):
            st.session_state['calculated'] = True
            st.session_state['mode'] = app_mode
            st.session_state['params'] = params
        else:
            st.error("Select ROI first.")

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
    
    # DL Layout is slightly different (needs wide map for results)
    if mode == "🤖 DL Water Segmentation":
        m = get_safe_map(700)
        col_map, col_res = st.columns([3, 1])
        
        with st.spinner("Initializing Deep Learning Engine..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_dl_model_from_drive(device=device)
            
            image = None
            profile = None
            transform = None
            crs = None
            bounds = None
            
            # A. LOAD DATA
            if p['source'] == 'upload':
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(p['file'].getbuffer())
                    tiff_path = tmp.name
                image, profile, transform, crs, bounds = read_geotiff(tiff_path)
                m.add_raster(tiff_path, layer_name="Uploaded Image", zoom_to_layer=True)
                
            elif p['source'] == 'pc':
                st.info("Querying Microsoft Planetary Computer...")
                # Convert GEE Geometry to GeoJSON dict
                roi_json = roi.getInfo() 
                result = build_planetary_computer_image_for_aoi(roi_json, p['sat_type'])
                if result[0] is None:
                    st.error("No cloud-free imagery found.")
                    st.stop()
                image, profile, transform, crs, bounds, count = result
                st.toast(f"Composited {count} images from Planetary Computer")
                
                # Save temp to visualize on map
                with tempfile.NamedTemporaryFile(suffix="_pc.tif", delete=False) as tmp_pc:
                    with rasterio.open(tmp_pc.name, 'w', **profile) as dst:
                        dst.write(image)
                    m.add_raster(tmp_pc.name, layer_name="Satellite Composite", zoom_to_layer=True)

            # B. INFERENCE
            st.info("Running U-Net Inference...")
            mask, prob = predict_large_image(model, image, device=device)
            
            # C. VECTORIZE
            st.info("Vectorizing Results...")
            gdf = mask_to_vector(mask, transform, crs)
            
            # D. VISUALIZE
            style = {"color": "#00BFFF", "weight": 2, "fillOpacity": 0.5, "fillColor": "#00BFFF"}
            if not gdf.empty:
                m.add_gdf(gdf, layer_name="Detected Water", style=style, info_mode="on_hover")
            else:
                st.warning("No water bodies detected.")

            with col_map:
                m.to_streamlit()
                
            with col_res:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">📊 DL RESULTS</div>', unsafe_allow_html=True)
                if not gdf.empty:
                    st.metric("Water Bodies", len(gdf))
                    st.metric("Total Area", f"{gdf['area_km2'].sum():.2f} km²")
                    st.dataframe(gdf[['id', 'area_km2']], hide_index=True, use_container_width=True)
                    
                    # Download Shapefile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        shp_path = os.path.join(tmpdir, "water_mask.shp")
                        gdf.to_file(shp_path)
                        zip_path = os.path.join(tmpdir, "water_mask.zip")
                        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                            for ext in ["shp", "shx", "dbf", "prj"]:
                                fpath = os.path.join(tmpdir, f"water_mask.{ext}")
                                if os.path.exists(fpath): zf.write(fpath, arcname=os.path.basename(fpath))
                        with open(zip_path, "rb") as f:
                            st.download_button("📥 Download Shapefile", f.read(), "water_mask.zip", "application/zip", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Standard GEE Modules Layout
    else:
        col_map, col_res = st.columns([3, 1])
        m = get_safe_map(700)
        m.centerObject(roi, 13)
        image_to_export = None 
        vis_export = {}

        # ==========================================
        # LOGIC A: RWH SITE SUITABILITY
        # ==========================================
        if mode == "📍 RWH Site Suitability":
            with st.spinner("Calculating Hydrological Suitability..."):
                
                # --- 1. DATA ACQUISITION & NORMALIZATION ---
                
                # A. Precipitation (CHIRPS)
                rain = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")\
                    .filterDate(p['start'], p['end']).select('precipitation').mean().clip(roi)
                # Normalize rain min-max within ROI for relative difference
                min_max_rain = rain.reduceRegion(ee.Reducer.minMax(), roi, 1000).getInfo()
                r_min = min_max_rain.get('precipitation_min', 0)
                r_max = min_max_rain.get('precipitation_max', 100)
                rain_n = rain.unitScale(r_min, r_max)

                # B. Topography (Slope & TWI)
                dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(roi)
                slope = ee.Terrain.slope(dem)
                
                # Calculate TWI (Topographic Wetness Index) for Flow Accumulation
                flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').clip(roi)
                # TWI formula approx: ln(a / tan(b))
                twi = flow_acc.log().subtract(slope.multiply(0.01745).tan().log()).rename('twi')
                
                # Normalize TWI (Higher is better for collection)
                twi_n = twi.unitScale(2, 12).clamp(0, 1) # Clamp outliers
                
                # Normalize Slope (Lower is generally better, but not 0)
                slope_n = ee.Image(1).subtract(slope.clamp(0, 25).unitScale(0, 25))

                # Combined Terrain Score (50% TWI, 50% Slope)
                terrain_score = twi_n.add(slope_n).divide(2)

                # C. LULC (ESA WorldCover)
                lulc = ee.Image("ESA/WorldCover/v100/2020").select('Map').clip(roi)
                # Remap: Ag(40)/Grass(30)/Scrub(20) = High. Forest(10) = Med. Urban(50)/Water(80) = Mask.
                lulc_score = lulc.remap(
                    [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], 
                    [0.5, 0.8, 0.9, 1.0, 0.0, 0.6, 0.1, 0.0, 0.1, 0.1, 0.1]
                ).rename('lulc_score')
                
                # Strict Mask for exclusion (Urban & Water)
                exclusion_mask = lulc.neq(50).And(lulc.neq(80)).And(slope.lt(20))

                # D. Soil (Clay Content -> Retention)
                try:
                    soil = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').mean().clip(roi)
                    soil_n = soil.unitScale(0, 60) # Normalize clay 0-60%
                except:
                    soil_n = ee.Image(0.5).clip(roi)

                # --- 2. WEIGHTED OVERLAY (MCDA) ---
                tot = p['total'] if p['total'] > 0 else 1
                
                suitability = (
                    rain_n.multiply(p['w_rain']/tot)
                    .add(terrain_score.multiply(p['w_slope']/tot))
                    .add(lulc_score.multiply(p['w_lulc']/tot))
                    .add(soil_n.multiply(p['w_soil']/tot))
                )
                
                # Apply Masks
                final_suitability = suitability.updateMask(exclusion_mask).clip(roi)
                
                # Visualization
                vis = {'min': 0.3, 'max': 0.8, 'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']}
                
                m.addLayer(dem, {'min':0, 'max':1000, 'palette':['black','white']}, 'DEM (Hidden)', False)
                m.addLayer(final_suitability, vis, 'RWH Suitability Index')
                m.add_colorbar(vis, label="RWH Potential (MCDA Score)")
                
                image_to_export = final_suitability
                vis_export = vis

                # Find Peak Suitability Points (Potential Check Dams)
                try:
                    high_suit = final_suitability.gt(0.75)
                    vectors = high_suit.reduceToVectors(
                        geometry=roi, scale=100, geometryType='centroid', 
                        eightConnected=False, maxPixels=1e8
                    )
                    m.addLayer(vectors, {'color': 'blue'}, 'Suggested Locations')
                except: pass

                with col_res:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-label">📊 CLASSIFICATION</div>', unsafe_allow_html=True)
                    
                    # Classify 1-5
                    classes = ee.Image(0).where(final_suitability.lt(0.3), 1)\
                        .where(final_suitability.gte(0.3).And(final_suitability.lt(0.5)), 2)\
                        .where(final_suitability.gte(0.5).And(final_suitability.lt(0.65)), 3)\
                        .where(final_suitability.gte(0.65).And(final_suitability.lt(0.8)), 4)\
                        .where(final_suitability.gte(0.8), 5).updateMask(exclusion_mask).clip(roi)
                    
                    df = calculate_area_by_class(classes, roi, 30)
                    name_map = {"Class 1": "Very Low", "Class 2": "Low", "Class 3": "Moderate", "Class 4": "High", "Class 5": "Optimal"}
                    if not df.empty:
                        df['Class'] = df['Class'].map(name_map).fillna(df['Class'])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.warning("No suitable areas found within thresholds.")
                    st.markdown("</div>", unsafe_allow_html=True)

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
        elif mode == "Flood Extent Mapping":
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
                        date_pre = ee.Date(before_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                        date_post = ee.Date(after_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()

                        before = before_col.median().clip(roi)
                        after = after_col.mosaic().clip(roi)
                        
                        smoothing = 50
                        before_f = before.focal_mean(smoothing, 'circle', 'meters')
                        after_f = after.focal_mean(smoothing, 'circle', 'meters')
                        
                        difference = after_f.divide(before_f)
                        difference_binary = difference.gt(p['threshold'])
                        
                        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                        occurrence = gsw.select('occurrence')
                        permanent_water_mask = occurrence.gt(30)
                        
                        flooded = difference_binary.updateMask(permanent_water_mask.Not())
                        
                        dem = ee.Image('WWF/HydroSHEDS/03VFDEM')
                        slope = ee.Algorithms.Terrain(dem).select('slope')
                        flooded = flooded.updateMask(slope.lt(5))
                        
                        flooded = flooded.updateMask(flooded.connectedPixelCount().gte(8))
                        flooded = flooded.selfMask()
                        
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

        # ==========================================
        # LOGIC D: WATER QUALITY (Sentinel-2)
        # ==========================================
        elif mode == "🧪 Water Quality":
            with st.spinner(f"Computing {p['param']} (Scientific Mode)..."):
                try:
                    # 1. PRE-PROCESSING FUNCTION (Improved Masking)
                    def mask_clouds_and_water(img):
                        # Cloud Masking (using S2_CLOUD_PROBABILITY)
                        cloud_prob = ee.Image(img.get('cloud_mask')).select('probability')
                        is_cloud = cloud_prob.gt(p['cloud'])
                        
                        # Scale Bands to Reflectance (0 to 1)
                        bands = img.select(['B.*']).multiply(0.0001)
                        
                        # Water Masking (NDWI > 0.0) 
                        ndwi = bands.normalizedDifference(['B3', 'B8']).rename('ndwi')
                        is_water = ndwi.gt(0.0)
                        
                        return bands.updateMask(is_cloud.Not()).updateMask(is_water).copyProperties(img, ['system:time_start'])

                    # 2. LOAD COLLECTIONS
                    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(p['start'], p['end']).filterBounds(roi)
                    s2_cloud = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(p['start'], p['end']).filterBounds(roi)
                    
                    # Join collections
                    s2_joined = ee.Join.saveFirst('cloud_mask').apply(
                        primary=s2_sr, secondary=s2_cloud,
                        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
                    )
                    
                    processed_col = ee.ImageCollection(s2_joined).map(mask_clouds_and_water)
                    
                    # 3. COMPUTE SCIENTIFIC INDICES
                    viz_params = {}
                    result_layer = None
                    layer_name = ""
                    
                    if "Turbidity" in p['param']:
                        def calc_ndti(img):
                            ndti = img.normalizedDifference(['B4', 'B3']).rename('value')
                            return ndti.copyProperties(img, ['system:time_start'])
                        
                        final_col = processed_col.map(calc_ndti)
                        result_layer = final_col.mean().clip(roi)
                        viz_params = {'min': -0.15, 'max': 0.15, 'palette': ['0000ff', '00ffff', 'ffff00', 'ff0000']} 
                        layer_name = "Turbidity Index (NDTI)"

                    elif "TSS" in p['param']:
                        def calc_tss(img):
                            tss = img.expression('2950 * (b4 ** 1.357)', {'b4': img.select('B4')}).rename('value')
                            return tss.copyProperties(img, ['system:time_start'])
                        
                        final_col = processed_col.map(calc_tss)
                        result_layer = final_col.median().clip(roi)
                        viz_params = {'min': 0, 'max': 50, 'palette': ['0000ff', '00ffff', 'ffff00', 'ff0000', '5c0000']}
                        layer_name = "TSS (Est. mg/L)"

                    elif "Cyanobacteria" in p['param']:
                        def calc_cyano(img):
                            cyano = img.expression('b5 / b4', {
                                'b5': img.select('B5'), 'b4': img.select('B4')
                            }).rename('value')
                            return cyano.copyProperties(img, ['system:time_start'])
                        
                        final_col = processed_col.map(calc_cyano)
                        result_layer = final_col.max().clip(roi)
                        viz_params = {'min': 0.8, 'max': 1.5, 'palette': ['0000ff', '00ff00', 'ff0000']}
                        layer_name = "Cyano Risk (Ratio > 1)"

                    elif "Chlorophyll" in p['param']:
                        def calc_ndci(img):
                            ndci = img.normalizedDifference(['B5', 'B4']).rename('value')
                            return ndci.copyProperties(img, ['system:time_start'])
                        
                        final_col = processed_col.map(calc_ndci)
                        result_layer = final_col.mean().clip(roi)
                        viz_params = {'min': -0.1, 'max': 0.2, 'palette': ['0000ff', '00ffff', '00ff00', 'ff0000']}
                        layer_name = "Chlorophyll-a (NDCI)"

                    elif "CDOM" in p['param']:
                        def calc_cdom(img):
                            cdom = img.expression('b3 / b2', {
                                'b3': img.select('B3'), 'b2': img.select('B2')
                            }).rename('value')
                            return cdom.copyProperties(img, ['system:time_start'])
                        
                        final_col = processed_col.map(calc_cdom)
                        result_layer = final_col.median().clip(roi)
                        viz_params = {'min': 0.5, 'max': 2.0, 'palette': ['0000ff', 'yellow', 'brown']}
                        layer_name = "CDOM Proxy (Green/Blue)"

                    # 4. VISUALIZATION
                    if result_layer:
                        image_to_export = result_layer
                        vis_export = viz_params
                        m.addLayer(result_layer, viz_params, layer_name)
                        m.add_colorbar(viz_params, label=layer_name)

                        # 5. CHARTING
                        with col_res:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="card-label">📈 TREND ANALYSIS</div>', unsafe_allow_html=True)
                            try:
                                def get_stats(img):
                                    date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
                                    val = img.reduceRegion(
                                        reducer=ee.Reducer.median(), 
                                        geometry=roi, 
                                        scale=20, 
                                        maxPixels=1e9
                                    ).values().get(0)
                                    return ee.Feature(None, {'date': date, 'value': val})
                                
                                fc = final_col.map(get_stats).filter(ee.Filter.notNull(['value']))
                                data_list = fc.reduceColumns(ee.Reducer.toList(2), ['date', 'value']).get('list').getInfo()
                                
                                if data_list:
                                    df_chart = pd.DataFrame(data_list, columns=['Date', 'Value'])
                                    df_chart['Date'] = pd.to_datetime(df_chart['Date'])
                                    df_chart = df_chart.sort_values('Date').dropna()
                                    
                                    st.area_chart(df_chart, x='Date', y='Value', color="#005792")
                                    st.caption(f"Median {layer_name} over time")
                                    
                                    # Export Data CSV
                                    csv = df_chart.to_csv(index=False).encode('utf-8')
                                    st.download_button("Download CSV", csv, "water_quality_ts.csv", "text/csv")
                                else:
                                    st.warning("No clear water pixels found (Try reducing cloud threshold).")

                            except Exception as e:
                                st.warning(f"Chart Error: {e}")
                            st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

        # --- COMMON EXPORT TOOLS ---
        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">📥 EXPORTS</div>', unsafe_allow_html=True)
            
            if st.button("Save to Drive (GeoTIFF)"):
                if image_to_export:
                    desc = f"GeoSarovar_{mode.split(' ')[1]}_{datetime.now().strftime('%Y%m%d')}"
                    ee.batch.Export.image.toDrive(
                        image=image_to_export, description=desc,
                        scale=30, region=roi, folder='GeoSarovar_Exports'
                    ).start()
                    st.toast("Export started! Check Google Drive.")
                else:
                    st.warning("No result to export.")

            st.markdown("---")
            report_title = st.text_input("Report Title", f"Analysis: {mode}")
            if st.button("Generate Map Image"):
                with st.spinner("Rendering..."):
                    if image_to_export:
                        # Determine visualization type
                        is_cat = False
                        c_names = None
                        cmap = None
                        
                        if mode == "Flood Extent Mapping":
                            is_cat = True; c_names = ['Flood Extent']
                        elif mode == "⚠️ Encroachment (S1 SAR)": 
                            is_cat = True; c_names = ['Stable Water', 'Encroachment', 'New Water']
                        elif 'palette' in vis_export:
                            cmap = vis_export['palette']
                        
                        buf = generate_static_map_display(image_to_export, roi, vis_export, report_title, cmap_colors=cmap, is_categorical=is_cat, class_names=c_names)
                        if buf:
                            st.download_button("Download JPG", buf, "GeoSarovar_Map.jpg", "image/jpeg", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_map:
            m.to_streamlit()
