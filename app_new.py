import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import rasterio
import geoai
import tempfile
import logging
from typing import Dict, Optional
import json
from pathlib import Path
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "PANEL_EFFICIENCY": 0.2,
    "ELECTRICITY_RATE": 0.17,  # $/kWh for california
    "INSTALLATION_COST_PER_KW": 3000,  # Rs per kW in california
    "CO2_PER_KWH": 0.225,  # kg CO2/kWh for California
    "CO2_PER_TREE": 22,  # kg CO2 per tree per year
    "DETECTION_PARAMS": {
        "confidence_threshold": 0.4,
        "mask_threshold": 0.5,
        "min_object_area": 100,
        "overlap": 0.25,
        "chip_size": (400, 400),
        "batch_size": 4
    }
}

@st.cache_resource
def load_detector():
    """Cache the GeoAI detector to avoid reloading"""
    try:
        return geoai.SolarPanelDetector()
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        st.error("Failed to initialize solar panel detector. Please check your installation.")
        return None

@st.cache_data
def load_geotiff_cached(_self):
    """Cache the GeoTIFF loading results"""
    try:
        if not os.path.exists(_self.geotiff_path):
            st.error(f"GeoTIFF file not found at: {_self.geotiff_path}")
            return None
            
        with rasterio.open(_self.geotiff_path) as src:
            image = src.read()
            if image.shape[0] > 3:
                image = image[:3]
            image = np.transpose(image, (1, 2, 0))
            return image
            
    except Exception as e:
        logger.error(f"Error loading GeoTIFF file: {e}")
        st.error(f"Failed to load satellite imagery: {str(e)}")
        return None

@st.cache_data
def detect_solar_panels_cached(_self, image):
    """Cache solar panel detection results"""
    try:
        output_path = os.path.join(_self.temp_dir, "solar_panel_masks.tif")
        
        with st.spinner("Detecting solar panels..."):
            masks_path = _self.detector.generate_masks(
                _self.geotiff_path,
                output_path=output_path,
                **CONFIG["DETECTION_PARAMS"],
                verbose=False
            )

            gdf = geoai.orthogonalize(
                input_path=masks_path,
                output_path=os.path.join(_self.temp_dir, "solar_panels.geojson"),
                epsilon=0.2
            )

            gdf = geoai.add_geometric_properties(gdf)
            return gdf[(gdf["elongation"] < 10) & (gdf["area_m2"] > 5)]

    except Exception as e:
        logger.error(f"Error in solar panel detection: {e}")
        st.error(f"Failed to detect solar panels: {str(e)}")
        return None

class AdvancedSolarDetector:
    def __init__(self):
        """Initialize the detector with GeoAI"""
        self.detector = load_detector()
        if not self.detector:
            raise RuntimeError("Failed to initialize detector")
            
        self.irradiance_db = self.connect_to_irradiance_database()
        self.temp_dir = tempfile.mkdtemp()
        self.geotiff_path = os.getenv("GEOTIFF_PATH", "C:\solarinsight\solar_panels_davis_ca.tif")

    def connect_to_irradiance_database(self) -> Dict:
        """Simulate a solar irradiance database."""
        return {
            "ahmedabad": {
                "avg_daily_irradiance": 5.5,
                "annual_solar_potential": 1825,
                "peak_sun_hours": 5.8,
            }
        }
    
    def load_geotiff(self):
        """Load the GeoTIFF file using cached function"""
        return load_geotiff_cached(self)

    def detect_solar_panels(self, image):
        """Detect solar panels using cached function"""
        return detect_solar_panels_cached(self, image)

    def get_closest_irradiance_data(self, latitude: float, longitude: float) -> Dict:
        """Get irradiance data for the closest known location."""
        # This is a simple implementation - you might want to make it more sophisticated
        # by finding the closest location based on actual distance
        return self.irradiance_db["ahmedabad"]

    def fetch_nasa_power_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch solar radiation data from NASA POWER API.
        
        Args:
            latitude (float): Latitude of the area.
            longitude (float): Longitude of the area.
            
        Returns:
            dict: Solar radiation data.
        """
        try:
            # NASA POWER API endpoint
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            
            # Calculate date range for the last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Format dates for API
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # Request parameters
            params = {
                "parameters": "ALLSKY_SFC_SW_DWN",  # Surface solar radiation
                "community": "RE",  # Renewable Energy community
                "longitude": longitude,
                "latitude": latitude,
                "start": start_str,
                "end": end_str,
                "format": "JSON"
            }
            
            st.info("Fetching solar radiation data from NASA POWER API...")
            
            # Make the request
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                # Process the data to extract annual averages
                if "properties" in data and "parameter" in data["properties"]:
                    radiation_data = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
                    # Calculate average daily radiation (kWh/m²/day)
                    daily_values = list(radiation_data.values())
                    avg_radiation = sum(float(v) for v in daily_values if v != -999) / len(daily_values)
                    # Convert from MJ/m²/day to kWh/m²/day (1 MJ = 0.277778 kWh)
                    avg_radiation_kwh = avg_radiation * 0.277778
                    
                    return {
                        "avg_daily_irradiance": round(avg_radiation_kwh, 2),
                        "annual_solar_potential": round(avg_radiation_kwh * 365, 2),
                        "peak_sun_hours": round(avg_radiation_kwh, 1),
                        "source": "NASA POWER"
                    }
            
            # If the API request fails, use the closest location from our database
            st.warning("Couldn't fetch NASA POWER data. Using simulated data instead.")
            return self.get_closest_irradiance_data(latitude, longitude)
            
        except Exception as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            st.error(f"Error fetching NASA POWER data: {e}")
            return self.get_closest_irradiance_data(latitude, longitude)

    def analyze_solar_potential(self, latitude: float, longitude: float) -> Optional[Dict]:
        """Analyze solar potential using the downloaded GeoTIFF."""
        try:
            with st.spinner("Loading satellite imagery..."):
                satellite_image = self.load_geotiff()
            
            if satellite_image is not None:
                st.subheader("🛰️ Source Imagery")
                st.image(satellite_image, caption="Tiff File")

                with st.spinner("Detecting solar panels..."):
                    gdf = self.detect_solar_panels(satellite_image)

                if gdf is not None and not gdf.empty:
                    # Display interactive map with progress bar
                    with st.spinner("Generating interactive map..."):
                        m = geoai.view_vector_interactive(
                            gdf, 
                            style_kwds={"color": "red", "fillOpacity": 0.3},
                            tiles=self.geotiff_path
                        )
                        st.components.v1.html(m._repr_html_(), height=500)

                    # Calculate statistics
                    total_area = gdf["area_m2"].sum()
                    total_panels = len(gdf)
                    
                    # Get irradiance data from NASA POWER API
                    irradiance_data = self.fetch_nasa_power_data(latitude, longitude)
                    
                    # Calculate solar potential
                    total_capacity = total_area * CONFIG["PANEL_EFFICIENCY"]
                    
                    return {
                        "total_panels": total_panels,
                        "total_area": total_area,
                        "panel_areas": gdf["area_m2"].tolist(),
                        "total_capacity_kW": round(total_capacity, 2),
                        "daily_energy_potential_kWh": round(total_capacity * irradiance_data["peak_sun_hours"], 2),
                        "annual_energy_potential_kWh": round(total_capacity * irradiance_data["annual_solar_potential"], 2),
                        "local_irradiance": irradiance_data
                    }
                else:
                    st.warning("No solar panels detected in the area.")
            return None
        except Exception as e:
            logger.error(f"Error in solar potential analysis: {e}")
            st.error(f"Analysis failed: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
            st.error(f"Failed to clean up temporary files: {str(e)}")

def display_results(results: Dict):
    """Display analysis results in a structured way."""
    st.success("✅ Analysis Complete!")
    
    # Show metrics in columns
    cols = st.columns(3)
    metrics = [
        ("Total Solar Panels", results["total_panels"]),
        ("Total Area", f"{results['total_area']:.1f} m²"),
        ("Total Capacity", f"{results['total_capacity_kW']} kW"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    # Display insights with tooltips
    st.subheader("📊 Solar Potential Insights")
    insights_df = pd.DataFrame([
        ["Daily Energy Potential", f"{results['daily_energy_potential_kWh']} kWh", 
         "Estimated daily energy generation potential"],
        ["Annual Energy Potential", f"{results['annual_energy_potential_kWh']} kWh",
         "Estimated annual energy generation potential"],
        ["Solar Irradiance", f"{results['local_irradiance']['avg_daily_irradiance']} kWh/m²/day",
         "Average daily solar radiation in the area"]
    ], columns=["Metric", "Value", "Description"])
    st.dataframe(insights_df, use_container_width=True)
    
    # Financial Analysis
    st.subheader("💰 Financial Analysis")
    installation_cost = results['total_capacity_kW'] * CONFIG["INSTALLATION_COST_PER_KW"]
    annual_savings = results['annual_energy_potential_kWh'] * CONFIG["ELECTRICITY_RATE"]
    payback_years = installation_cost / annual_savings if annual_savings > 0 else 0
    
    fin_cols = st.columns(3)
    with fin_cols[0]:
        st.metric("Installation Cost", f"₹{installation_cost:,.0f}")
    with fin_cols[1]:
        st.metric("Annual Savings", f"₹{annual_savings:,.0f}")
    with fin_cols[2]:
        st.metric("Payback Period", f"{payback_years:.1f} years")
    
    # Environmental Impact
    st.subheader("🌍 Environmental Impact")
    co2_reduction = results['annual_energy_potential_kWh'] * CONFIG["CO2_PER_KWH"]
    trees_equivalent = co2_reduction / CONFIG["CO2_PER_TREE"]
    
    env_cols = st.columns(2)
    with env_cols[0]:
        st.metric("Annual CO2 Reduction", f"{co2_reduction:,.0f} kg")
    with env_cols[1]:
        st.metric("Equivalent Trees", f"{trees_equivalent:.0f} trees")

def main():
    st.set_page_config(
        page_title="Solar Potential Analyzer",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add sidebar with information
    with st.sidebar:
        st.title("About")
        st.info("""
        This tool analyzes solar potential for a given location using satellite imagery.
        It detects solar panels and calculates potential energy generation, financial benefits,
        and environmental impact.
        """)
        
        st.title("Settings")
        st.checkbox("Show detailed analysis", value=True, key="show_details")
    
    st.title("☀️ Solar Potential Analyzer for Shyamal, Ahmedabad")
    
    try:
        # Initialize detector
        detector = AdvancedSolarDetector()
        
        # Fixed coordinates for Shyamal, Ahmedabad
        latitude = 38.5449
        longitude = -121.7405
        
        # Analyze button
        if st.button("🚀 Analyze Solar Potential"):
            with st.spinner("Starting analysis..."):
                results = detector.analyze_solar_potential(latitude, longitude)
                if results:
                    display_results(results)
                
                # Cleanup
                detector.cleanup()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()