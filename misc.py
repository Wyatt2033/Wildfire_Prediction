# Unused code, pasted from other files

def main():
    merge_data = pd.read_csv('datasets/merged_data.csv')

    state = st.selectbox('Select a state', state_names, key='state',
                         index=None)
    if state:
        state = geocoding.state_get_abrev(state)
        counties = data.get_counties_for_state(state)
        for county in counties:
            lat, long = geocoding.get_lat_long(county, state)
            with open("output.txt", "a") as f:
                f.write(f'{county}, {state}: {lat}, {long}\n')
            weather_data = weather.get_weather_data(lat, long)
            weather_data_averages = pd.DataFrame(weather_data.mean()).transpose()
            pd.set_option('display.max_columns', 7)
            # print(weather_data_averages)
            with open("output.txt", "a") as f:
                f.write(f'{weather_data}\n')
            wildfire_risk = randomForest.predict_wildfire_risk(weather_data_averages)
            risk = "High" if wildfire_risk[0] else "Low"
            st.write(f"{county}, {state}: {risk} risk of wildfire")
            print(f"{county}, {state}: {risk} risk of wildfire; Probability: {wildfire_risk[1]}")