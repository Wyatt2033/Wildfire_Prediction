"""
Module for Wildfire Risk Prediction
"""
import logging
import os
import joblib
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

# List of state names
STATE_NAMES = ["Alabama", "Arkansas", "Arizona", "California",
               "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky",
               "Louisiana", "Massachusetts", "Maryland", "Maine",
               "Michigan", "Minnesota", "Missouri", "Mississippi",
               "Montana", "North Carolina", "North Dakota", "Nebraska",
               "New Hampshire", "New Jersey", "New Mexico", "Nevada",
               "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota",
               "Tennessee", "Texas", "Utah", "Virginia", "Vermont",
               "Washington", "Wisconsin", "West Virginia", "Wyoming"]

LAST_RUN_DATES = {}
LAST_RUN_DATE = None
LAST_RUN_FILE = 'last_run.pkl'
DAYS_TO_UPDATE = 7
DATASET_PATH = 'datasets/uscounties.csv'
GDF = gpd.read_file('./map_data/cb_2018_us_county_5m.shp')






#print(gdf.columns)


class DataUpdater:
    def __init__(self):
        self.last_run_date = self.load_last_run_date()

    @staticmethod
    def load_last_run_date():
        """
        Load the last run date from the file
        """
        if os.path.exists(LAST_RUN_FILE):
            return joblib.load(LAST_RUN_FILE)
        else:
            updater.download_and_unpack_cache()
            updater.download_model_from_google_drive()
            return None

    def data_age_check(self):
        """
        Checks if the data is older than a week and updates it if necessary.
        """
        # Get the current date
        current_date = datetime.date.today()

        # If the function has not been run in the last 7 days
        if self.last_run_date is None or (current_date - self.last_run_date).days > DAYS_TO_UPDATE:
            logging.info('Updating weather data...')
            updater.update_weather_data()
            # Update the last run date
            self.last_run_date = current_date
            joblib.dump(self.last_run_date, LAST_RUN_FILE)

# Calls load_data() from data.py
# (Not used with current implementation)
def load_data():
    """
    Calls load_data() from data.py
    """
    print('Loading Data')
    (weather_train_data, weather_test_data, weather_validation_data,
     wildfire_train_data, wildfire_test_data, wildfire_validation_data) = \
        data.load_data()
    return (weather_train_data, weather_test_data,
            weather_validation_data, wildfire_train_data,
            wildfire_test_data, wildfire_validation_data)


# Calls merge_data() from data.py
def merge_data(weather_train_data, wildfire_train_data,
               weather_test_data, wildfire_test_data,
               weather_validation_data,
               wildfire_validation_data):
    print('Merging Data')
    print(wildfire_train_data.shape)
    merged_train_data, merged_val_data, merged_test_data = \
        data.merge_data(weather_train_data, wildfire_train_data,
                        weather_test_data, wildfire_test_data,
                        weather_validation_data,
                        wildfire_validation_data)
    return merged_train_data, merged_val_data, merged_test_data


# Calls split_data() from randomForest.py
def split_data(merged_data):
    """
    Calls split_data from randomForest.py
    """
    print('Splitting Data')
    x_train, x_test, y_train, y_test = (
        randomForest.split_data(merged_data))
    return x_train, x_test, y_train, y_test


# Calls train_model() from randomForest.py
def train_model(x_train, y_train, x_test, y_test):
    """
    Calls train_model() from randomForest.py
    """
    print('Training Model')
    model = randomForest.train_model(x_train, y_train)
    randomForest.save_model(model)
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    return model




def get_average_weather(county_name, state_name):
    """
    Fetches the average weather data for a given county and state.

    Parameters:
    county_name (str): The name of the county.
    state_name (str): The name of the state.

    Returns:
    dict: A dictionary containing the average weather data.
    """
    # Use more descriptive variable names
    dataset = pd.read_csv(DATASET_PATH)
    latitude_longitude = geocoding.get_lat_long(dataset, county_name, state_name)

    if latitude_longitude is None:
        raise ValueError(f"Could not find latitude and longitude for {county_name}, {state_name}")

    latitude, longitude = latitude_longitude
    weather_data = weather.get_cached_weather_data(latitude, longitude)
    recent_weather_data = weather_data.tail(30)

    average_weather_data = {
        'Average max temperature': f"{round(recent_weather_data['temperature_2m_max'].mean(), 1)} °C",
        'Average minimum temperature': f"{round(recent_weather_data['temperature_2m_min'].mean(), 1)} °C",
        'Average precipitation': f"{round(recent_weather_data['precipitation_sum'].mean(), 1)} mm",
        'Average max wind speed': f"{round(recent_weather_data['wind_speed_10m_max'].mean(), 1)} km/h",
        'Average max wind gust speed': f"{round(recent_weather_data['wind_gusts_10m_max'].mean(), 1)} km/h",
        'Average surface pressure': f"{round(recent_weather_data['surface_pressure'].mean(), 1)} hPa",
        'Average humidity': f"{round(recent_weather_data['relative_humidity_2m'].mean(), 1)}%",
        'Average dew point': f"{round(recent_weather_data['dew_point_2m'].mean(), 1)} °C"
    }

    return average_weather_data




def main():
    """
    Main function of the application
    """
    st.write("""

    # Wildfire Risk Prediction

    This is a web app to predict the risk of wildfires in the United States.

        """)

    updater.download_and_unpack_cache()
    updater.download_model_from_google_drive()
    # data_age_check()
    merge_data = pd.read_csv('./datasets/merged_data.csv')
    accuracy = randomForest.print_accuracy()
    st.write(f'The accuracy of the wildfire risk condition prediction is'
             f' {accuracy * 100:.2f}%')
    st.write(
        'The prediction accuracy percentage  is based on the '
        'proportion of correct prdictions (true positives and true '
        'negatives) among the total wildfire test cases examined.')
    st.write('This is an overall view of the contiguous United States. '
             'The map shows counties with conditions that '
             'favor wildfires.')
    geocoding.country_fire_map()
    #update_weather_data()
    st.write('For a more detailed view of a state, please select'
             ' one below.')
    state = st.selectbox('Select a state', STATE_NAMES, key='state',
                         index=None)
    if state:
        state = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state)
        counties = [geocoding.standardize_county_name(county)
                    for county in counties]
        geocoding.plot_fire_map(state, counties)
        st.write(f'For a daily average weather data for a specific county '
                 f'in {state}, please select one below.')
        county = st.selectbox('Select a county', counties, key='county'
                              , index=None)
        if county:
            geocoding.get_chart_data(county, state)
            average_temperature = get_average_weather(county, state)
            st.write(f'The daily average weather data for {county}, {state} is:')
            for variable, value in average_temperature.items():
                st.write(f'{variable}: {value}')


# Calls main() when the script is run
if __name__ == "__main__":
    main()
