import os
import threading
import time

import joblib
import schedule
import datetime
import data
import geocoding
import randomForest
import update_scheduler
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
print(gdf.columns)


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


def update_weather_data():
    start_index = state_names.index("Alabama")
    for state in state_names[start_index:]:
        state_abrev = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state_abrev)
        print("Getting weather data for", state_abrev)
        for county in counties:
            try:
                county = geocoding.standardize_county_name(county)
                df = pd.read_csv('datasets/uscounties.csv')
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
                last_update_dates[county] = datetime.now()

            except Exception as e:
                print(f"An error occurred while updating weather data for {county}, {state}: {e}")
        break
    update_state_cache()


def update_state_cache():
    start_index = state_names.index("Alabama")
    for state in state_names[start_index:]:

        state_abrev = geocoding.state_get_abrev(state)
        map_filename = f'./cache/state_data_cache/{state_abrev}_map_data.joblib'
        if os.path.exists(map_filename):
            os.remove(map_filename)

        new_data = geocoding.generate_map_data(state_abrev)
        joblib.dump(new_data, map_filename)


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




def listen():
    global last_run_date
    # Get the current date
    current_date = datetime.date.today()
    # If the function has not been run today
    if last_run_date != current_date:
        print('Updating weather data...')
        update_weather_data()
        # Update the last run date
        last_run_date = current_date



def main():
    accuracy = randomForest.print_accuracy()
    st.write(f'The accuracy of the wildfire risk condition prediction is {accuracy * 100:.2f}%')
    merge_data = pd.read_csv('./datasets/merged_data.csv')
    model_file_path = './models/trained_model.pkl'
    if not os.path.exists(model_file_path):
        print('Model not found. Downloading model...')
        update_scheduler.download_file_from_google_drive()
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
        st.write(f'For a 30-day average weather data for a specific county in {state}, please select one below.')
        county = st.selectbox('Select a county', counties, key='county', index=None)
        if county:
            geocoding.get_chart_data(county, state)
            average_temperature = get_average_weather(county, state)
            st.write(f'The 30-day average weather data for {county}, {state} is:')
            for variable, value in average_temperature.items():
                st.write(f'{variable}: {value}')

# Calls main() when the script is run
if __name__ == "__main__":
    main()
