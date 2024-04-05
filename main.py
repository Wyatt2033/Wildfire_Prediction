import os
import time

import schedule

import data
import geocoding
import randomForest
import update_scheduler
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta


state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado",
               "Connecticut",  "Delaware", "Florida", "Georgia", "Hawaii",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland",
               "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota",
               "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
               "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
               "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

st.write("""

# Wildfire Risk Prediction

This is a web app to predict the risk of wildfires in the United States.

    """)

gdf = gpd.read_file('map_data/cb_2018_us_county_5m.shp')
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
    start_index = state_names.index("Alaska")
    for state in state_names[start_index:]:
        state_abrev = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state_abrev)
        print("Getting weather data for", state_abrev)
        for county in counties:
            try:
                county = geocoding.standardize_county_name(county)
                lat_long = geocoding.get_lat_long(county, state)
                if lat_long is None:
                    print(f"Could not find latitude and longitude for {county}, {state}")
                    continue
                lat, long = lat_long
                last_update_date = last_update_dates.get(county)

                if last_update_date is not None and datetime.now() - last_update_date < timedelta(weeks=1):
                    print(f"Skipping update for {county}, {state} as the data is newer than a week")
                    continue

                weather_data = weather.get_cached_weather_data(lat, long)
                last_update_dates[county] = datetime.now()
            except Exception as e:
                print(f"An error occurred while updating weather data for {county}, {state}: {e}")


# Calls main() from main.py

def main():
    merge_data = pd.read_csv('datasets/merged_data.csv')
    model_file_path = 'models/trained_model.pkl'
    if not os.path.exists(model_file_path):
        print('Model not found. Downloading model...')
        update_scheduler.download_file_from_google_drive()

    state = st.selectbox('Select a state', state_names, key='state',
                         index=None)
    if state:
        state = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state)
        counties = [geocoding.standardize_county_name(county) for county in counties]
        geocoding.plot_fire_map(state, counties)

    schedule.every().monday. do(update_weather_data)

    while True:
        schedule.run_pending()
        time.sleep(1)






# Calls main() when the script is run
if __name__ == "__main__":
    main()