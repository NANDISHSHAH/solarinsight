import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import folium
from PIL import Image
from selenium import webdriver
import os


class AdvancedSolarDetector:
    def __init__(self):
        # Load pre-trained models
        self.roof_detection_model = self.load_model("roof_detection_model.h5", "Roof Detection")
        self.solar_panel_model = self.load_model("solar_panel_model.h5", "Solar Panel Detection")

        # Simulated solar irradiance data
        self.irradiance_db = self.connect_to_irradiance_database()

    def load_model(self, model_path, model_name):
        """Load pre-trained deep learning models."""
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
            return None

    def connect_to_irradiance_database(self):
        """Simulate a solar irradiance database."""
        return {
            "ahmedabad": {
                "avg_daily_irradiance": 5.5,  # kWh/m²/day
                "annual_solar_potential": 1825,  # kWh/m²/year
                "peak_sun_hours": 5.8,
            }
        }

    def fetch_satellite_imagery(self, latitude, longitude, zoom=18):
        """
        Fetch high-resolution imagery using OpenStreetMap with folium and selenium.
        
        Args:
            latitude (float): Latitude of the area.
            longitude (float): Longitude of the area.
            zoom (int): Zoom level for imagery.

        Returns:
            np.ndarray or None: Satellite image as numpy array.
        """
        try:
            # Create OSM map with Folium
            map_center = [latitude, longitude]
            map_object = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap")

            # Save map as HTML
            map_path = "map.html"
            map_object.save(map_path)

            # Convert HTML to image using Selenium
            image_path = "map_image.png"
            self.html_to_image(map_path, image_path)

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            st.error(f"Error fetching satellite imagery: {e}")
            return None

    def html_to_image(self, html_path, image_path):
        """
        Convert HTML to image using Selenium.
        
        Args:
            html_path (str): Path to the saved HTML file.
            image_path (str): Path to save the resulting image.
        """
        # Set up a headless browser using Selenium
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        file_path = f"file://{os.path.abspath(html_path)}"
        driver.get(file_path)

        # Set window size and take screenshot
        driver.set_window_size(800, 600)
        driver.save_screenshot(image_path)
        driver.quit()

    def detect_roofs_and_panels(self, satellite_image):
        """Detect roofs and solar panels from satellite imagery."""
        if satellite_image is None:
            return None

        preprocessed_image = self.preprocess_image(satellite_image)
        st.image(
    (preprocessed_image[0] * 255).astype("uint8"),
    caption="Preprocessed Image",
    channels="RGB"
)
        # Roof detection
        roof_mask = self.roof_detection_model.predict(preprocessed_image)
        panel_mask = self.solar_panel_model.predict(preprocessed_image)

        total_roofs = np.sum(roof_mask > 0.5)
        total_panels = np.sum(panel_mask > 0.5)

        return {
            "total_roofs": int(total_roofs),
            "total_panels": int(total_panels),
            "roof_coverage_percentage": float(total_roofs / preprocessed_image.shape[1] * 100),
            "panel_coverage_percentage": float(total_panels / preprocessed_image.shape[1] * 100),
        }

    def preprocess_image(self, image):
        """Preprocess image for the model."""
        resized_image = cv2.resize(image, (256, 256))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)

    def analyze_solar_potential(self, latitude, longitude):
        """Analyze solar potential using OSM satellite imagery."""
        satellite_image = self.fetch_satellite_imagery(latitude, longitude)

        # Detect roofs and panels
        detection_results = self.detect_roofs_and_panels(satellite_image)

        # Fetch local irradiance data
        irradiance_data = self.irradiance_db.get("ahmedabad", {})

        if detection_results and irradiance_data:
            panel_capacity = 0.4  # kW per panel
            total_capacity = detection_results["total_panels"] * panel_capacity

            return {
                **detection_results,
                "total_capacity_kW": round(total_capacity, 2),
                "daily_energy_potential_kWh": round(total_capacity * irradiance_data["peak_sun_hours"], 2),
                "annual_energy_potential_kWh": round(total_capacity * irradiance_data["annual_solar_potential"] / 365, 2),
                "local_irradiance": irradiance_data,
            }
        return None


# Streamlit App
def main():
    st.set_page_config(page_title="Advanced Solar Potential Analyzer", page_icon="☀️", layout="wide")
    st.title("☀️ Advanced Solar Potential Analyzer")
    st.markdown(
        """
        Analyze the solar potential of any region using:
        - ✅ AI-powered roof and solar panel detection.
        - ✅ Free OpenStreetMap satellite imagery.
        - ✅ Local solar irradiance data.
    """
    )

    # Initialize the detector
    detector = AdvancedSolarDetector()

    # Get user input
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Enter Latitude", value=23.0337, format="%.4f")  # Ahmedabad default
    with col2:
        longitude = st.number_input("Enter Longitude", value=72.5220, format="%.4f")

    # Analyze button
    if st.button("🚀 Analyze Solar Potential"):
        with st.spinner("Analyzing solar potential..."):
            results = detector.analyze_solar_potential(latitude, longitude)

            if results:
                st.success("✅ Analysis Complete!")
                # Show metrics
                cols = st.columns(3)
                metrics = [
                    ("Total Roofs", results["total_roofs"]),
                    ("Solar Panels", results["total_panels"]),
                    ("Total Capacity", f"{results['total_capacity_kW']} kW"),
                ]
                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)

                # Display insights
                st.subheader("📊 Solar Insights")
                insights_df = pd.DataFrame(
                    [
                        ["Roof Coverage", f"{results['roof_coverage_percentage']:.2f}%"],
                        ["Panel Coverage", f"{results['panel_coverage_percentage']:.2f}%"],
                        ["Daily Energy Potential", f"{results['daily_energy_potential_kWh']} kWh"],
                        ["Annual Energy Potential", f"{results['annual_energy_potential_kWh']} kWh"],
                        ["Local Solar Irradiance", f"{results['local_irradiance'].get('avg_daily_irradiance', 'N/A')} kWh/m²/day"],
                    ],
                    columns=["Metric", "Value"],
                )
                st.dataframe(insights_df, use_container_width=True)

                # Display the map image
                st.subheader("🗺️ Satellite Image")
                st.image("map_image.png", caption="Satellite Image of Selected Location")

            else:
                st.error("⚠️ Unable to complete analysis. Please check your inputs.")


if __name__ == "__main__":
    main()
