import io
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.geocoders import GoogleV3
from matplotlib.colors import ListedColormap

import randomForest
import weather



def create_legend():
    fig, ax = plt.subplots(figsize=(6, 6))
    red_patch = mpatches.Patch(color='red', label='High risk')
    green_patch = mpatches.Patch(color='green', label='Low risk')
    yellow_patch = mpatches.Patch(color='white', label='Not assessed')

    fig, ax = plt.subplots()
    legend = plt.legend(handles=[red_patch, green_patch, yellow_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05))
    ax.axis('off')
    st.pyplot(fig)
def plot_fire_map(state, counties):
    create_legend()
    state_fips = {
        "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12",
        "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
        "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
        "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
        "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
    }



    plot_placeholder = st.empty()
    gdf = gpd.read_file('map_data/cb_2018_us_county_5m.shp')
    gdf_state = gdf[gdf['STATEFP'] == state_fips[state]]
    gdf_state['NAME'] = gdf_state['NAME'].apply(standardize_county_name)
    gdf_state['risk_color'] = 'white'
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_facecolor('#0b0e12')
    fig.patch.set_facecolor('#0b0e12')
    for county in counties:
        centroid = gdf_state.loc[gdf_state['NAME'].str.contains(county), 'geometry'].centroid
        ax.annotate(county, (centroid.x, centroid.y), color='black', fontsize=4, ha='center')
    for i, county in enumerate(counties):
        lat, long = get_lat_long(county, state)
        weather_data = weather.get_weather_data(lat, long)
        weather_data_averages = pd.DataFrame(weather_data.mean()).transpose()
        wildfire_risk = randomForest.predict_wildfire_risk(weather_data_averages)
        st.write(f"{county}, {state}: {wildfire_risk} risk of wildfire")
        risk = "High" if wildfire_risk else "Low"

        risk_color = 'red' if wildfire_risk[0] else 'green'

        gdf_state.loc[gdf_state['NAME'].str.contains(county), 'risk_color'] = risk_color
        centroid = gdf_state.loc[gdf_state['NAME'].str.contains(county), 'geometry'].centroid

        gdf_state.plot(ax=ax, color='yellow', edgecolor='black')  # Add edgecolor parameter here
        gdf_state.plot(ax=ax, color=gdf_state['risk_color'], edgecolor='black')  # Add edgecolor parameter here
        if i % 3 == 0:
            plot_placeholder.pyplot(fig)

        try:
            cmap = ListedColormap(['yellow', 'red', 'green'])
            gdf_state.plot(ax=ax, color='yellow')
            gdf_state.plot(ax=ax, color=gdf_state['risk_color'])
            gdf_state.plot(ax=ax, color='yellow', edgecolor='black')  # Add edgecolor parameter here
            gdf_state.plot(ax=ax, color=gdf_state['risk_color'], edgecolor='black')  # Add edgecolor parameter here

            gdf_state.loc[gdf_state['NAME'].str.contains(county), 'risk_color'] = risk_color

            ax.set_aspect('equal')

        except ValueError:
            ax.set_aspect('auto')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption='Wildfire Risk Map', use_column_width=True)


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
