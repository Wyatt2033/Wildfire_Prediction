import tarfile
import gdown
import os
from multiprocessing import Pool
from joblib import load
import joblib
import datetime
import data
import geocoding
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd

from main import last_update_dates

state_names = ["Alabama", "Arkansas", "Arizona", "California", "Colorado",
               "Connecticut", "Delaware", "Florida", "Georgia",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
               "Maryland",
               "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina",
               "North Dakota",
               "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
               "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
               "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]


def download_model_from_google_drive():
    model_file_path = './models/trained_model.pkl'
    if not os.path.exists(model_file_path):
        print('Model not found. Downloading model...')
        file_id = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
        destination = "./models/trained_model.pkl"
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=True)


def download_and_unpack_cache():
    # Ensure the cache directory exists
    if not os.path.exists('./cache'):
        os.makedirs('./cache')

    # Google Drive file id
    file_id = "1LzpeMkfnjxXZrKy7FxSiFVCq6mAoKKoh"
    # Destination path
    destination = "./cache/archive.tar.gz"
    # Google Drive download link
    url = f"https://drive.google.com/uc?id={file_id}"
    # Download the file
    gdown.download(url, destination, quiet=True)

    # Unpack the .tar.gz file
    with tarfile.open(destination, 'r:gz') as tar:
        tar.extractall(path='./cache')


def download_weather_from_google_drive():
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    file_id = "1zNO2r4daZrEZB3pYtNHS14FYqal_GtMc"
    destination = "./cache/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=True)


def download_state_from_google_drive():
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    file_id = "1vPj4S__ZK-cv17kfFvZ4oU_rVgAWMLpL"
    destination = "./cache/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=True)


def update_state_cache():
    start_index = state_names.index("Alabama")
    for state in state_names[start_index:]:

        state_abrev = geocoding.state_get_abrev(state)
        map_filename = f'./cache/state_data_cache/{state_abrev}_map_data.joblib'
        if os.path.exists(map_filename):
            os.remove(map_filename)

        new_data = geocoding.generate_map_data(state_abrev)
        joblib.dump(new_data, map_filename)


def update_state_weather_cache():
    state_fips = {
        "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
        "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
        "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
        "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
        "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
    }
    for state in state_names:
        state_abrev = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state_abrev)
        map_filename = f'./cache/weather_cache/{state_abrev}_map_data.csv'

        list_placeholder = st.empty()
        plot_placeholder = st.empty()
        gdf = gpd.read_file('./map_data/cb_2018_us_county_5m.shp')
        gdf_state = gdf[gdf['STATEFP'] == state_fips[state_abrev]]
        gdf_state['NAME'] = gdf_state['NAME'].apply(geocoding.standardize_county_name)
        gdf_state['risk_color'] = 'white'
        gdf_state = gdf_state.copy()
        gdf_states = []
        with Pool() as p:
            gdf_states = p.map(geocoding.predict_and_plot,
                               [(county, state, gdf_state.copy()) for county in counties])

        gdf_state = pd.concat([gdf_state] + gdf_states)
        joblib.dump(gdf_state, map_filename)


def update_weather_data():
    start_index = state_names.index("Alabama")
    for state in state_names[start_index:]:
        state_abrev = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state_abrev)
        print("Getting weather_cache.json data for", state_abrev)
        for county in counties:
            print(f"Getting weather data for {county}, {state_abrev}")
            try:
                county = geocoding.standardize_county_name(county)
                df = pd.read_csv('./datasets/uscounties.csv')
                lat_long = geocoding.get_lat_long(df, county, state_abrev)
                if lat_long is None:
                    print(f"Could not find latitude and longitude for {county}, {state}")
                    continue
                lat, long = lat_long
                last_update_date = last_update_dates.get(county)

                if last_update_date is not None and datetime.datetime.now() - last_update_date < datetime.timedelta(
                        weeks=1):
                    print(f"Skipping update for {county}, {state} as the data is newer than a week")
                    continue

                weather_data = weather.get_cached_weather_data(lat, long)
                last_update_dates[county] = datetime.datetime.now()

            except Exception as e:
                print(f"An error occurred while updating weather data for {county}, {state}: {e}")
        break

    update_state_weather_cache()
    update_state_cache()
