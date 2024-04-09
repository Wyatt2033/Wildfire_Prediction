import contextlib
import io
import os
import warnings

import joblib

import data
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
from joblib import dump, load
from multiprocessing import Pool


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
    green_patch = mpatches.Patch(color='green', label='Low Risk')
    ax.legend(handles=[red_patch, green_patch], loc='upper right')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption='Legend', use_column_width=True)


def predict_and_plot(args):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    county, state, gdf_state = args
    try:
        lat, long = get_lat_long(county, state)
        if lat is None or long is None:
            print(f"Latitude and longitude not found for {county}, {state}")
            return
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            weather_data = weather.get_cached_weather_data(lat, long)

        weather_data_averages = pd.DataFrame(weather_data.mean()).transpose()
        wildfire_risk = randomForest.predict_wildfire_risk(weather_data_averages)
        risk_color = 'red' if wildfire_risk[0] else 'green'
        if gdf_state['NAME'].str.contains(county).any():
            gdf_state.loc[
                gdf_state['NAME'].str.contains(county), 'risk_color'] = risk_color
            #print(f"Assigned color {risk_color} for {county}, {state}")
        else:
            print(f"County name {county} does not match in the GeoDataFrame")
        # Check for duplicate rows in the gdf_state DataFrame
        duplicate_rows = gdf_state[gdf_state.duplicated(['NAME'], keep=False)]

        if not duplicate_rows.empty:
            print("Duplicate rows in gdf_state:")
            print(duplicate_rows)
        return gdf_state.loc[gdf_state['NAME'].str.contains(county) & gdf_state['risk_color'].notna()]


    except Exception as e:
        print(f"An error occurred while processing {county}, {state}: {e}")
        gdf_state.loc[gdf_state['NAME'].str.contains(county), 'risk_color'] = 'orange'

        # Only return the modified row for the county
        return gdf_state.loc[gdf_state['NAME'].str.contains(county) & gdf_state['risk_color'].notna()]

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
    map_filename = f'./map_data/{state}_map_data.csv'
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
        gdf = gpd.read_file('./map_data/cb_2018_us_county_5m.shp')
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
        green_patch = Line2D([0], [0], color='green', linewidth=5, label='Low Risk')
        legend = SizedLegend(ax, [red_patch, green_patch], ['High Risk', 'Low Risk'], loc='upper right', frameon=True,
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
                            color='blue',
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
        if isinstance(buf, io.BytesIO):
            st.image(buf, caption='Wildfire Risk Map', use_column_width=True)
        else:
            print("Error: buf does not contain image data")

    st.image(buf, caption='Wildfire Risk Map', use_column_width=True)


def country_fire_map():
    # List of all state abbreviations
    state_abrevs = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN", "IA", "KS",
                    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
                    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
                    "WI", "WY"]

    # Initialize an empty list to store the GeoDataFrames
    gdfs = []
    print("Generating map data for all states")
    # Loop over each state abbreviation
    for state_abrev in state_abrevs:
        print(f"Generating map data for {state_abrev}")
        # Construct the filename of the cache file
        filename = f'./map_data/{state_abrev}_map_data.joblib'

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
    #print(gdf_us)
    minx, miny, maxx, maxy = -125, 24, -66, 50

    # Exclude outliers
    gdf_us = gdf_us.cx[minx:maxx, miny:maxy]
    # Plot the combined GeoDataFrame as a heatmap
    cmap = ListedColormap(['green', 'white', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    gdf_us.plot(column='risk_color', ax=ax, legend=True, cmap=cmap)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    gdf_us.plot(column='risk_color', ax=ax, legend=True, cmap=cmap)

    ax.set_facecolor('#0b0e12')

    fig.patch.set_facecolor('#0b0e12')

    ax.set_xlim(gdf_us.geometry.bounds.minx.min(), gdf_us.geometry.bounds.maxx.max())
    ax.set_ylim(gdf_us.geometry.bounds.miny.min(), gdf_us.geometry.bounds.maxy.max())

    plt.axis('off')
    # ax.set_position([0, 0, 1, 1])
    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=True)


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
    gdf = gpd.read_file('./map_data/cb_2018_us_county_5m.shp')

    # Filter the GeoDataFrame for the specific state
    gdf_state = gdf[gdf['STATEFP'] == state_fips[state_abrev]]

    # Standardize the county names
    gdf_state['NAME'] = gdf_state['NAME'].apply(standardize_county_name)

    # Initialize the risk color column
    #gdf_state['risk_color'] = 'orange'

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

    gdf_states = [predict_and_plot((county, state_abrev, gdf_state.copy())) for county in counties]
    print("County names after prediction:")
    print(len(counties))
    print(f"2 Data for {state_abrev}:")
    print(gdf_state['risk_color'].value_counts())
    # Concatenate the results
    gdf_state = pd.concat(gdf_states)
    # Check for counties that did not get a risk color assigned
    print(f"3 Data for {state_abrev}:")
    print(gdf_state['risk_color'].value_counts())

    missing_colors = gdf_state[gdf_state['risk_color'].isna()]
    if not missing_colors.empty:
        print("Counties missing risk color:")
        print(missing_colors['NAME'])
        print(len(missing_colors))
        # Print out the county names in the GeoDataFrame
        print("County names in GeoDataFrame:")
        print(gdf_state['NAME'])
        print(len(gdf_state['NAME']))
    return gdf_state


def get_lat_long(county_name, state_name):
    geolocator = GoogleV3(api_key='AIzaSyBxIbGubpa41aTqVXdpFSzHfzaYibiXe6M')
    location = geolocator.geocode(f"{county_name}, {state_name}")

    return (location.latitude, location.longitude)


def standardize_county_name(county_name):
    suffixes = [" County", " Parish", " Borough", " City", " Municipality", " Census Area", " Area", " and"]
    for suffix in suffixes:
        if suffix in county_name:
            county_name = county_name.replace(suffix, "")
            county_name = county_name.strip()
            county_name = county_name.replace('county', '')

    return county_name


def state_get_abrev(state):
    state_abrev = {
        "Alabama": "AL",
        "Alaska": "AK",
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
