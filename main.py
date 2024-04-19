import os
import threading
import time
from multiprocessing import Pool

from joblib import load
import joblib
import schedule
import datetime
import data
import geocoding
import randomForest
import updater
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.metrics import accuracy_score

state_names = ["Alabama", "Arkansas", "Arizona", "California", "Colorado",
               "Connecticut", "Delaware", "Florida", "Georgia",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland",
               "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota",
               "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
               "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
               "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

st.write("""

# Wildfire Risk Prediction

This is a web app to predict the risk of wildfires in the United States.

    """)

gdf = gpd.read_file('./map_data/cb_2018_us_county_5m.shp')


#print(gdf.columns)


# Calls load_data() from data.py (Not used with current implementation)
def load_data():
    print('Loading Data')
    weather_train_data, weather_test_data, weather_validation_data, wildfire_train_data, wildfire_test_data, wildfire_validation_data = data.load_data()
    return weather_train_data, weather_test_data, weather_validation_data, wildfire_train_data, wildfire_test_data, wildfire_validation_data


# Calls merge_data() from data.py (Not used with current implementation)
def merge_data(weather_train_data, wildfire_train_data, weather_test_data, wildfire_test_data, weather_validation_data,
               wildfire_validation_data):
    print('Merging Data')
    print(wildfire_train_data.shape)
    merged_train_data, merged_val_data, merged_test_data = data.merge_data(weather_train_data, wildfire_train_data,
                                                                           weather_test_data, wildfire_test_data,
                                                                           weather_validation_data,
                                                                           wildfire_validation_data)
    return merged_train_data, merged_val_data, merged_test_data


# Calls split_data() from randomForest.py
def split_data(merged_data):
    print('Splitting Data')
    x_train, x_test, y_train, y_test = randomForest.split_data(merged_data)
    return x_train, x_test, y_train, y_test


# Calls train_model() from randomForest.py
def train_model(x_train, y_train, x_test, y_test):
    print('Training Model')
    model = randomForest.train_model(x_train, y_train)
    randomForest.save_model(model)
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    return model


last_update_dates = {}

last_run_date = None


def get_average_weather(county, state):
    df = pd.read_csv('datasets/uscounties.csv')
    lat_long = geocoding.get_lat_long(df, county, state)
    if lat_long is None:
        print(f"Could not find latitude and longitude for {county}, {state}")
        return None
    lat, long = lat_long
    weather_data = weather.get_cached_weather_data(lat, long)
    last_30_days = weather_data.tail(30)
    average_weather = {
        'Average max temperature': f"{round(last_30_days['temperature_2m_max'].mean(), 1)} °C",
        'Average minimum temperature': f"{round(last_30_days['temperature_2m_min'].mean(), 1)} °C",
        'Average precipitation': f"{round(last_30_days['precipitation_sum'].mean(), 1)} mm",
        'Average max wind speed': f"{round(last_30_days['wind_speed_10m_max'].mean(), 1)} km/h",
        'Average max wind gust speed': f"{round(last_30_days['wind_gusts_10m_max'].mean(), 1)} km/h",
        'Average surface pressure': f"{round(last_30_days['surface_pressure'].mean(), 1)} hPa",
        'Average humidity': f"{round(last_30_days['relative_humidity_2m'].mean(), 1)}%",
        'Average dew point': f"{round(last_30_days['dew_point_2m'].mean(), 1)} °C"
    }
    return average_weather


def data_age_check():
    global last_run_date
    # Get the current date
    current_date = datetime.date.today()

    # Load the last run date from the file
    if os.path.exists('last_run_date.pkl'):
        last_run_date = joblib.load('last_run_date.pkl')

    else:
        updater.download_and_unpack_cache()
        updater.download_model_from_google_drive()

    # If the function has not been run in the last 7 days
    if last_run_date is None or (current_date - last_run_date).days > 7:
        print('Updating weather data...')
        updater.update_weather_data()
        # Update the last run date
        last_run_date = current_date
        joblib.dump(last_run_date, 'last_run_date.pkl')


def main():
    data_age_check()
    merge_data = pd.read_csv('./datasets/merged_data.csv')
    accuracy = randomForest.print_accuracy()
    st.write(f'The accuracy of the wildfire risk condition prediction is {accuracy * 100:.2f}%')
    st.write(
        'The prediction accuracy percentage  is based on the proportion of correct prdictions (true positives and true negatives) among the total wildfire test cases examined.')
    st.write('This is an overall view of the contiguous United States. The map shows counties with conditions that '
             'favor wildfires.')
    geocoding.country_fire_map()
    #update_weather_data()
    st.write('For a more detailed view of a state, please select one below.')
    state = st.selectbox('Select a state', state_names, key='state',
                         index=None)
    if state:
        state = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state)
        counties = [geocoding.standardize_county_name(county) for county in counties]
        geocoding.plot_fire_map(state, counties)
        st.write(f'For a daily average weather data for a specific county in {state}, please select one below.')
        county = st.selectbox('Select a county', counties, key='county', index=None)
        if county:
            geocoding.get_chart_data(county, state)
            average_temperature = get_average_weather(county, state)
            st.write(f'The daily average weather data for {county}, {state} is:')
            for variable, value in average_temperature.items():
                st.write(f'{variable}: {value}')


# Calls main() when the script is run
if __name__ == "__main__":
    main()
