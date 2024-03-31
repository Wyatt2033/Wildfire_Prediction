import data
import geocoding
import randomForest
import weather
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.metrics import accuracy_score


state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado",
               "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland",
               "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota",
               "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
               "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
               "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

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

# Calls main() from main.py

def main():
    merge_data = pd.read_csv('datasets/merged_data.csv')

    state = st.selectbox('Select a state', state_names, key='state',
                         index=None)
    if state:
        state = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state)
        counties = [geocoding.standardize_county_name(county) for county in counties]
        geocoding.plot_fire_map(state, counties)






# Calls main() when the script is run
if __name__ == "__main__":
    main()