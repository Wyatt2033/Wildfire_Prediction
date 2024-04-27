import logging
import tarfile
import gdown
import os
from multiprocessing import Pool
import joblib
import datetime
import data
import geocoding
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd

from main import LAST_RUN_DATES

STATE_NAMES = ["Alabama", "Arkansas", "Arizona", "California", "Colorado",
               "Connecticut", "Delaware", "Florida", "Georgia",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
               "Maryland",
               "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina",
               "North Dakota",
               "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
               "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
               "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]


DATASET_PATH = 'datasets/uscounties.csv'
LAST_RUN_DATES = {}
MODEL_FILE_PATH = 'models/trained_model.pkl'
CACHE_DIR = 'cache'
ARCHIVE_PATH = f'{CACHE_DIR}/archive.tar.gz'
MODEL_FILE_ID = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
ARCHIVE_FILE_ID = "1LzpeMkfnjxXZrKy7FxSiFVCq6mAoKKoh"
WEATHER_FILE_ID = "1zNO2r4daZrEZB3pYtNHS14FYqal_GtMc"
STATE_FILE_ID = "1vPj4S__ZK-cv17kfFvZ4oU_rVgAWMLpL"

class WeatherUpdater:
    def __init__(self):
        self.last_run_dates = LAST_RUN_DATES

    def update_weather_data(self):
        """
        Updates the weather data for all states and counties.
        """
        start_index = STATE_NAMES.index("Alabama")
        for state in STATE_NAMES[start_index:]:
            state_abrev = geocoding.state_get_abrev(state)
            counties = data.get_counties_for_state(state_abrev)
            logging.info(f"Getting weather_cache.json data for {state_abrev}")
            for county in counties:
                logging.info(f"Getting weather data for {county}, {state_abrev}")
                try:
                    county = geocoding.standardize_county_name(county)  # standardize the county name
                    df = pd.read_csv(DATASET_PATH)
                    lat_long = geocoding.get_lat_long(df, county, state_abrev)
                    if lat_long is None:
                        raise ValueError(f"Could not find latitude and longitude for {county}, {state}")
                    lat, long = lat_long
                    last_update_date = self.last_run_dates.get(county)

                    if last_update_date is not None and datetime.datetime.now() - last_update_date < datetime.timedelta(weeks=1):
                        logging.info(f"Skipping update for {county}, {state} as the data is newer than a week")
                        continue

                    weather_data = weather.get_cached_weather_data(lat, long)
                    self.last_run_dates[county] = datetime.datetime.now()

                except Exception as e:
                    logging.error(f"An error occurred while updating weather data for {county}, {state}: {e}")

        self.update_state_weather_cache()
        self.update_state_cache()

    def update_state_cache(self):
        """
        Updates the state cache for all states.
        """
        start_index = STATE_NAMES.index("Alabama")
        for state in STATE_NAMES[start_index:]:
            state_abrev = geocoding.state_get_abrev(state)
            map_filename = f'{CACHE_DIR}/state_data_cache/{state_abrev}_map_data.joblib'
            if os.path.exists(map_filename):
                os.remove(map_filename)

            new_data = geocoding.generate_map_data(state_abrev)
            joblib.dump(new_data, map_filename)

    def update_state_weather_cache(self):
        """
        Updates the weather cache for all states.
        """
        state_fips = {
            "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
            "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
            "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
            "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
            "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
            "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
        }
        for state in STATE_NAMES:
            state_abrev = geocoding.state_get_abrev(state)
            counties = data.get_counties_for_state(state_abrev)
            counties = [geocoding.standardize_county_name(county) for county in counties]
            map_filename = f'./cache/weather_cache/{state_abrev}_map_data.csv'

            if os.path.exists(map_filename):
                try:
                    os.remove(map_filename)
                    print(f"Removed {map_filename}")

                except Exception as e:
                    print(f"Error removing {map_filename}: {e}")

            list_placeholder = st.empty()
            plot_placeholder = st.empty()
            gdf = gpd.read_file('map_data/cb_2018_us_county_5m.shp')
            gdf_state = gdf[gdf['STATEFP'] == state_fips[state_abrev]]
            gdf_state['NAME'] = gdf_state['NAME'].apply(geocoding.standardize_county_name)
            gdf_state['risk_color'] = 'white'
            gdf_state = gdf_state.copy()
            gdf_states = []
            with Pool() as p:
                gdf_states = p.map(geocoding.predict_and_plot,
                                   [(county, state_abrev, gdf_state.copy()) for county in counties])

            gdf_state = pd.concat([gdf_state] + gdf_states)
            joblib.dump(gdf_state, map_filename)



def download_model_from_google_drive():
    model_file_path = 'models/trained_model.pkl'
    if not os.path.exists(model_file_path):
        print('Model not found. Downloading model...')
        file_id = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
        destination = "./models/trained_model.pkl"
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=True)


def download_model_from_google_drive():
    """
    Downloads the trained model from Google Drive.
    """
    if not os.path.exists(MODEL_FILE_PATH):
        logging.info('Model not found. Downloading model...')
        destination_dir = os.path.dirname(MODEL_FILE_PATH)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_FILE_PATH, quiet=True)

def download_and_unpack_cache():
    """
    Downloads and unpacks the cache from Google Drive.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    url = f"https://drive.google.com/uc?id={ARCHIVE_FILE_ID}"
    gdown.download(url, ARCHIVE_PATH, quiet=True)

    with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:
        tar.extractall(path=CACHE_DIR)

def download_weather_from_google_drive():
    """
    Downloads the weather data from Google Drive.
    """
    destination = f"{CACHE_DIR}/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={WEATHER_FILE_ID}"
    gdown.download(url, destination, quiet=True)

def download_state_from_google_drive():
    """
    Downloads the state data from Google Drive.
    """
    destination = f"{CACHE_DIR}/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={STATE_FILE_ID}"
    gdown.download(url, destination, quiet=True)




