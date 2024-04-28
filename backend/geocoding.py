import json
import logging

from flask import Flask, jsonify
from flask_cors import CORS
import contextlib
import io
import os
import warnings
import joblib
from geopy import Nominatim
import data
import geocoding
import randomForest
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.geocoders import GoogleV3
from matplotlib.colors import ListedColormap
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from joblib import load
from multiprocessing import Pool

logging.basicConfig(filename='flask.log', level=logging.DEBUG)
app = Flask(__name__)
CORS(app)

STATE_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18"'][', "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
}


class SizedLegend(Legend):
    def __init__(self, parent, handles, labels, *args, **kwargs):
        super().__init__(parent, handles, labels, *args, **kwargs)
        self.legendPatch.set_boxstyle("round,pad=0.02")


def create_legend():
    fig, ax = plt.subplots()
    ax.set_aspect(0.1)
    ax.axis('off')
    fig.patch.set_facecolor('#0b0e12')
    ax.set_facecolor('#0b0e12')
    red_patch = mpatches.Patch(color='red', label='High Risk')
    blue_patch = mpatches.Patch(color='blue', label='Low Risk')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption='Legend', use_column_width=True)

def standardize_county_name(county_name):
    """Standardizes county names by removing certain suffixes."""
    suffixes = [" County", " Parish", " Borough", " City", " Municipality", " Census Area", " Area", " and", "City",
                "city"]
    for suffix in suffixes:
        if suffix in county_name:
            county_name = county_name.replace(suffix, "")
            county_name = county_name.strip()
            county_name = county_name.replace('county', '')
    return county_name
def get_lat_long(df, county_name, state_name):
    """Fetches latitude and longitude for a given county and state."""
    county_name = geocoding.standardize_county_name(county_name)
    try:
        row = df[(df['county'].str.contains(county_name)) & (df['state_id'] == state_name)]
        return row['lat'].values[0], row['lng'].values[0]
    except KeyError:
        geolocator = GoogleV3(api_key='AIzaSyBxIbGubpa41aTqVXdpFSzHfzaYibiXe6M')
        location = geolocator.geocode(f"{county_name}, {state_name}")
        if location is None:
            return None, None
        return (location.latitude, location.longitude)

def predict_and_plot(args):
    """Predicts and plots wildfire risk for a given county and state."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    county, state, gdf_state = args
    try:
        df = pd.read_csv('datasets/uscounties.csv')
        lat, long = get_lat_long(df, county, state)
        if lat is None or long is None:
            return
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            weather_data = weather.get_cached_weather_data(lat, long)
        weather_data_averages = pd.DataFrame(weather_data.mean()).transpose()
        wildfire_risk = randomForest.predict_wildfire_risk(weather_data_averages)
        risk_color = 'red' if wildfire_risk[0] else 'blue'
        if gdf_state['NAME'].str.contains(county).any():
            gdf_state.loc[
                gdf_state['NAME'].str.contains(county), 'risk_color'] = risk_color
        return gdf_state.loc[gdf_state['NAME'].str.contains(county) & gdf_state['risk_color'].notna()]
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"An error occurred while processing {county}, {state}: {e}\n")
        return gdf_state


def plot_fire_map(state, counties):
    # create_legend()
    state_fips = {
        "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
        "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
        "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
        "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
        "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
    }
    # Check if the map for this state has been generated since the last weather update.
    map_filename = f'./cache/weather_cache/{state}_map_data.csv'
    if os.path.exists(map_filename):
        buf = load(map_filename)
        grid_layout = ""
        county_numbers = {county: i for i, county in enumerate(counties)}
        for county, number in county_numbers.items():

            grid_layout += f"| {county:<20}: {number:<5} "
            if (number + 1) % 6 == 0:
                grid_layout += "\n"
        list_placeholder = st.empty()
        list_placeholder.markdown(grid_layout)
        print(f"Data for {state}:")
        print(state)

    else:

        list_placeholder = st.empty()
        plot_placeholder = st.empty()
        gdf = gpd.read_file('map_data/cb_2018_us_county_5m.shp')
        gdf_state = gdf[gdf['STATEFP'] == state_fips[state]]
        gdf_state['NAME'] = gdf_state['NAME'].apply(standardize_county_name)
        gdf_state['risk_color'] = 'white'
        gdf_state = gdf_state.copy()
        gdf_states = []
        with Pool() as p:
            gdf_states = p.map(predict_and_plot, [(county, state, gdf_state.copy()) for county in counties])

        gdf_state = pd.concat([gdf_state] + gdf_states)
        joblib.dump(gdf_state, map_filename)

        print("Finished processing all counties")
        # Plot map
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        ax.set_facecolor('#0b0e12')
        fig.patch.set_facecolor('#0b0e12')

        red_patch = Line2D([0], [0], color='red', linewidth=5, label='High Risk')
        blue_patch = Line2D([0], [0], color='blue', linewidth=5, label='Low Risk')
        legend = SizedLegend(ax, [red_patch, blue_patch], ['High Risk', 'Low Risk'], loc='upper right', frameon=True,
                             handlelength=0.5, prop={'size': 8})
        ax.add_artist(legend)

        gdf_state.plot(ax=ax, color=gdf_state['risk_color'], edgecolor='black')  # Add edgecolor parameter here
        county_numbers = {county: i for i, county in enumerate(counties)}
        for county in counties:
            centroid = gdf_state.loc[gdf_state['NAME'].str.contains(county), 'geometry'].centroid
            # ax.annotate(county, (centroid.x.iloc[0], centroid.y.iloc[0]), color='black', fontsize=4, ha='center')
            matching_geometries = gdf_state.loc[gdf_state['NAME'].str.contains(county), 'geometry']

            if not matching_geometries.empty:
                representative_point = matching_geometries.unary_union.representative_point()
                offset = 0
                ax.annotate(county_numbers[county], (representative_point.x + offset, representative_point.y),
                            color='white',
                            fontsize=8, ha='left', va='center')
            else:
                print(f"Could not find representative point for {county}")
        grid_layout = ""
        for county, number in county_numbers.items():

            grid_layout += f"| {county:<20}: {number:<5} "
            if (number + 1) % 6 == 0:
                grid_layout += "\n"
        list_placeholder.markdown(grid_layout)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        joblib.dump(buf, map_filename)
        buf = joblib.load(map_filename)
        # Ensure that buf contains image data

    st.image(buf, caption='Wildfire Risk Map', use_column_width=True)



@app.route('/api/country_fire_map', methods=['GET'])
def country_fire_map():
    app.logger.info('Processing country_fire_map request')
    # List of all state abbreviations
    state_abrevs = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN", "IA", "KS",
                    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
                    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
                    "WI", "WY"]
    state_fips = {
        "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
        "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
        "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
        "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
        "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
    }
    gdfs = []
    for state_abrev in state_abrevs:
        print(f"Generating map data for {state_abrev}")
        # Construct the filename of the cache file
        filename = f'./cache/state_data_cache/{state_abrev}_map_data.joblib'

        # Check if the file exists
        if os.path.exists(filename):
            # Load the data from the cache file
            gdf = joblib.load(filename)


        else:
            # Generate the data required for the map
            gdf = generate_map_data(state_abrev)
            print(f"Data for {state_abrev}:")
            print(gdf['risk_color'].value_counts())
            # Save the data into a cache file
            joblib.dump(gdf, filename)

        # Append the GeoDataFrame to the list
        gdfs.append(gdf)

    # Concatenate all the GeoDataFrames into a single GeoDataFrame
    gdf_us = pd.concat(gdfs)
    print("Data for the entire country:")
    print(gdf_us['risk_color'].value_counts())
    white_counties = gdf_us.loc[gdf_us['risk_color'] == 'white', 'NAME']
    print(white_counties)

    gdf_us['state_fips'] = gdf_us['STATEFP']
    # Group the GeoDataFrame by state and calculate the centroid of each state
    gdf_us['centroid'] = gdf_us.geometry.centroid
    gdf_states = gdf_us.dissolve(by='STATEFP', aggfunc='first')



    gdf_us_json = json.loads(gdf_us.to_json)

    gdf_states_json = json.loads(gdf_states.to_json)
    map_data = {
        'us': gdf_us_json,
        'states': gdf_states_json
    }

    map_data = gdf_states_json
    return jsonify(map_data)


def generate_map_data(state_abrev):
    # Define the state FIPS codes
    state_fips = {
        "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
        "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
        "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
        "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
        "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
    }

    # Read the shapefile
    gdf = gpd.read_file('map_data/cb_2018_us_county_5m.shp')

    # Filter the GeoDataFrame for the specific state
    gdf_state = gdf[gdf['STATEFP'] == state_fips[state_abrev]]

    # Standardize the county names
    gdf_state['NAME'] = gdf_state['NAME'].apply(standardize_county_name)

    # Initialize the risk color column

    # Make a copy of the GeoDataFrame

    gdf_state['risk_color'] = 'white'

    # Get the counties for the state
    counties = data.get_counties_for_state(state_abrev)

    # Standardize the county names
    counties = [standardize_county_name(county) for county in counties]

    # Remove duplicate county names
    counties = list(set(counties))
    # Print out the county names after standardization
    print("County names after standardization:")
    print(len(counties))
    print(f"1 Data for {state_abrev}:")
    print(gdf_state['risk_color'].value_counts())
    # Run the prediction for all counties
    for county in counties:
        gdf_county = predict_and_plot((county, state_abrev, gdf_state.copy()))
        if gdf_county is not None:
            gdf_state.loc[gdf_state['NAME'].str.contains(county), 'risk_color'] = gdf_county['risk_color']

    # Identify the counties that are still marked as 'white'
    white_counties = gdf_state.loc[gdf_state['risk_color'] == 'white', 'NAME']

    # Run the prediction again for the 'white' counties
    for county in white_counties:
        gdf_county = predict_and_plot((county, state_abrev, gdf_state.copy()))
        if gdf_county is not None:
            gdf_state.loc[gdf_state['NAME'].str.contains(county), 'risk_color'] = gdf_county['risk_color']



    return gdf_state


def get_lat_long_geolocator(county_name, state_name):
    geolocator = GoogleV3(api_key='AIzaSyBxIbGubpa41aTqVXdpFSzHfzaYibiXe6M')
    location = geolocator.geocode(f"{county_name}, {state_name}")

    if location is None:
        print(f"Could not find latitude and longitude for {county_name}, {state_name}, using GoogleV3 geocoder.")
        print('Trying Nominatim geocoder')
        geolocator = Nominatim(user_agent="geo_locator")
        location = geolocator.geocode(f"{county_name}, {state_name}")
        return location.latitude, location.longitude

    if location is None:
        print(f"Could not find latitude and longitude for {county_name}, {state_name}, using NOMINATIM.")
        return None, None

    return (location.latitude, location.longitude)


def get_chart_data(county, state):
    df = pd.read_csv('datasets/uscounties.csv')
    lat_long =get_lat_long(df, county, state)
    if lat_long is None:
        print(f"Could not find latitude and longitude for {county}, {state}")
        return None
    lat, long = lat_long
    weather_data = weather.get_cached_weather_data(lat, long)
    last_30_days = weather_data.tail(30)
    plt.figure(figsize=(10, 5))
    plt.plot( weather_data['temperature_2m_max'], label='Max Temperature')
    plt.plot( weather_data['temperature_2m_min'], label='Min Temperature')
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.title('14 Day Temperature Trend')
    plt.legend()
    st.pyplot(plt)


def state_get_abrev(state):
    state_abrev = {
        "Alabama": "AL",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY"
    }
    return state_abrev[state]

if __name__ == "__main__":
   app.run(port=5000)
