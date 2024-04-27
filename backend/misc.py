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



def country_fire_map():
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
    # Print out the state abbreviation and its corresponding FIPS number
    grid_layout = ""
    i = 0
    for state_abrev, fips_number in state_fips.items():
        grid_layout += f"| {state_abrev:<6}: {fips_number:<6} "
        if (int(i) + 1) % 6 == 0:
            i = i + 1
            grid_layout += "\n"
    list_placeholder = st.empty()
    list_placeholder.markdown(grid_layout)



    # Initialize an empty list to store the GeoDataFrames
    gdfs = []
    print("Generating map data for all states")
    # Loop over each state abbreviation
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
    #print(gdf_us)
    minx, miny, maxx, maxy = -125, 24, -66, 50

    # Exclude outliers
    gdf_us = gdf_us.cx[minx:maxx, miny:maxy]
    # Plot the combined GeoDataFrame
    cmap = ListedColormap(['blue', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    gdf_us.plot(column='risk_color', ax=ax, legend=True, cmap=cmap)
    gdf_us.boundary.plot(ax=ax, color='black', linewidth=0.5)
    gdf_states.boundary.plot(ax=ax, color='black', linewidth=2)
    # Adjust the legend
    white_patch = Line2D([0], [0], color='blue', linewidth=5, label='Low Risk')
    red_patch = Line2D([0], [0], color='red', linewidth=5, label='High Risk')
    legend = plt.legend(handles=[white_patch, red_patch], loc='upper right', frameon=True, handlelength=0.5,
                        prop={'size': 14})
    #ax.add_artist(legend)

    for x, y, label in zip(gdf_states.centroid.x, gdf_states.centroid.y, gdf_states.index):
        ax.text(x, y, label, color='white', fontsize=24)
    ax.set_facecolor('#0b0e12')

    fig.patch.set_facecolor('#0b0e12')

    ax.set_xlim(gdf_us.geometry.bounds.minx.min(), gdf_us.geometry.bounds.maxx.max())
    ax.set_ylim(gdf_us.geometry.bounds.miny.min(), gdf_us.geometry.bounds.maxy.max())

    plt.axis('off')
    # ax.set_position([0, 0, 1, 1])
    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=True)

    map_data_json = generate_map_data()
    return map_data_json
