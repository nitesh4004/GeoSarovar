import streamlit as st
import ee
import geemap.foliumap as geemap
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stubble Watch", page_icon="🔥", layout="wide")

# --- AUTHENTICATION & INITIALIZATION ---
# This handles the GEE login. On first run, it will open a browser window to authenticate.
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔥 Stubble Watch Control")
st.sidebar.markdown("Monitoring Crop Burning in Punjab/Haryana")

# Date Selection
today = datetime.date.today()
analysis_date = st.sidebar.date_input("Select Analysis Date", today)

# Convert date to GEE format
start_date = analysis_date.strftime("%Y-%m-%d")
end_date = (analysis_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

# --- DATA FETCHING FUNCTIONS ---

def get_fire_data(start, end):
    """Fetches NASA FIRMS data (VIIRS 375m) for the selected date."""
    dataset = ee.ImageCollection('FIRMS') \
        .filterDate(start, end) \
        .select('T21') # Brightness temperature channel
    return dataset

def get_pollution_data(start, end):
    """Fetches Sentinel-5P NO2 data to visualize smog/smoke."""
    dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
        .filterDate(start, end) \
        .select('NO2_column_number_density') \
        .mean()
    return dataset

def get_wind_data():
    """Fetches real-time wind forecast from NOAA GFS to predict smoke direction."""
    # Grab the latest available forecast
    dataset = ee.ImageCollection('NOAA/GFS0p25') \
        .filterDate(datetime.date.today().strftime("%Y-%m-%d"), 
                   (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")) \
        .select(['u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground']) \
        .first()
    return dataset

def calculate_wind_direction_at_delhi(wind_image):
    """Calculates wind direction specifically for Delhi to generate alert."""
    delhi_point = ee.Geometry.Point([77.1025, 28.7041]) # Delhi Coordinates
    
    # specific_wind = wind_image.reduceRegion(
    #     reducer=ee.Reducer.mean(),
    #     geometry=delhi_point,
    #     scale=25000 # GFS resolution is coarse (~27km)
    # ).getInfo()
    
    # Getting values safely
    try:
        # Note: GFS data might have a lag. If today's data isn't ready, this might be empty.
        # For a robust app, we'd add a fallback to yesterday's data.
        stats = wind_image.reduceRegion(reducer=ee.Reducer.first(), geometry=delhi_point, scale=27000).getInfo()
        u = stats.get('u_component_of_wind_10m_above_ground')
        v = stats.get('v_component_of_wind_10m_above_ground')
        
        if u is not None and v is not None:
            import math
            # Calculate wind direction (meteorological degrees)
            wind_deg = (180 / math.pi) * math.atan2(u, v) + 180
            return wind_deg
    except:
        return None
    return None

# --- APP LAYOUT ---

st.title(f"🛰️ Stubble Watch: Crop Burning Tracker")
st.markdown(f"**Analysis for: {analysis_date}**")

col1, col2 = st.columns([3, 1])

with col1:
    # 1. CREATE MAP
    m = geemap.Map(center=[30.5, 76.0], zoom=7) # Centered on Punjab
    m.add_basemap("CartoDB.DarkMatter") # Dark map makes fire pop

    # 2. ADD LAYERS
    
    # A. Sentinel-5P NO2 (Pollution Overlay)
    no2_layer = get_pollution_data(start_date, end_date)
    no2_vis = {
        'min': 0,
        'max': 0.0002,
        'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
    }
    m.addLayer(no2_layer, no2_vis, 'NO2 Pollution (Sentinel-5P)', True, 0.5) # 0.5 opacity

    # B. Fire Data (Red Dots)
    fire_collection = get_fire_data(start_date, end_date)
    fire_vis = {'min': 325, 'max': 400, 'palette': ['red', 'orange', 'yellow']}
    m.addLayer(fire_collection, fire_vis, 'Active Fires (FIRMS)')

    # C. Region of Interest (Punjab/Haryana Outline)
    # Adding a simple box or feature for context usually helps, 
    # but here we rely on the basemap labels.

    # Render Map in Streamlit
    m.to_streamlit(height=600)

with col2:
    st.subheader("⚠️ Smoke Alert System")
    
    # Get Wind Data from GEE
    wind_img = get_wind_data()
    wind_dir = calculate_wind_direction_at_delhi(wind_img)
    
    if wind_dir:
        st.metric(label="Wind Direction (at Delhi)", value=f"{int(wind_dir)}°")
        
        # Logic: If wind is from NW (approx 270-360 degrees), smoke comes to Delhi
        if 270 <= wind_dir <= 360 or 0 <= wind_dir <= 20:
            st.error("🚨 **CRITICAL ALERT**")
            st.write("Wind is blowing from Punjab/Haryana towards Delhi.")
            st.write("Expect severe smog in NCR within 24 hours.")
        else:
            st.success("✅ **SAFE ZONE**")
            st.write("Wind is blowing away from Delhi.")
            st.write("Smoke impact on NCR is currently low.")
    else:
        st.warning("Wind data unavailable for this date.")

    st.markdown("---")
    st.subheader("Data Sources")
    st.caption("• **Fires:** NASA FIRMS (VIIRS 375m)")
    st.caption("• **Air Quality:** Sentinel-5P (NO2)")
    st.caption("• **Wind:** NOAA GFS Forecast")

# --- ADD EXPLANATION ---
with st.expander("ℹ️ How to read this map"):
    st.write("""
    1. **Red/Yellow Dots:** These are active fires detected by NASA satellites in the last 24 hours. Brighter colors = hotter fires.
    2. **Colorful Haze:** This represents NO2 density (pollution). If you see red/yellow clouds over the red dots, that is the smoke plume forming.
    3. **Wind Alert:** The panel on the right calculates if that smoke is heading towards the capital.
    """)
