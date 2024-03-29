import pandas as pd


# load the data
def merge_weather(csv_files):
    # Create an empty list to store the DataFrames
    df_list = []

    # Loop through the list of CSV files and read each one into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all the DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('datasets/combined_weather.csv', index=False)


def pre_process_merge(data_csv, prep_weather_csv):
    data_csv = pd.read_csv(data_csv)
    prep_weather_csv = pd.read_csv(prep_weather_csv)
    wildfire_drop_columns = ['Shape', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                             'NWCG_REPORTING_UNIT_ID',
                             'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME',
                             'LOCAL_FIRE_REPORT_ID',
                             'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME', 'ICS_209_PLUS_INCIDENT_JOIN_ID',
                             'ICS_209_PLUS_COMPLEX_JOIN_ID', 'MTBS_ID', 'MTBS_FIRE_NAME', 'COMPLEX_NAME',
                             'NWCG_CAUSE_CLASSIFICATION', 'NWCG_GENERAL_CAUSE',
                             'NWCG_CAUSE_AGE_CATEGORY', 'CONT_DATE', 'CONT_DOY', 'DISCOVERY_TIME',
                             'OWNER_DESCR', 'FOD_ID', 'FPA_ID', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'CONT_DATE',
                             'CONT_DOY', 'CONT_TIME',
                             'FIRE_SIZE_CLASS', 'STATE', 'COUNTY', 'FIPS_NAME']
    weather_drop_columns = ['WS10M_MIN', 'WS50M_MIN', 'WS50M', 'WS50M_RANGE', 'WS50M_MAX', 'T2MWET']
    data_csv = data_csv.drop(columns=wildfire_drop_columns, axis=1)
    data_csv.to_csv('datasets/prep_wildfire.csv', index=False)
    prep_weather_csv = prep_weather_csv.drop(columns=weather_drop_columns, axis=1)
    prep_weather_csv['DATE'] = pd.to_datetime(prep_weather_csv['DATE'])

    prep_weather_csv = prep_weather_csv[prep_weather_csv['DATE'] >= '2015-01-01']
    prep_weather_csv.to_csv('datasets/prep_weather.csv', index=False)


def merge_data(prep_wildfire, prep_weather):
    prep_wildfire = prep_wildfire.rename(columns={'DISCOVERY_DATE': 'DATE'})

    prep_wildfire = prep_wildfire.dropna(subset=['FIPS_CODE'])

    merged_data = pd.merge(prep_wildfire, prep_weather, on=['DATE', 'FIPS_CODE'], how='inner')
    merged_data.to_csv('datasets/fire_merged_data.csv', index=False)


def load_data():
    weather_train_data = pd.read_csv('datasets/weather_train.csv')
    weather_test_data = pd.read_csv('datasets/weather_test.csv')
    weather_validation_data = pd.read_csv('datasets/weather_val.csv')
    wildfire_train_data = pd.read_csv('datasets/wildfire_train.csv')
    wildfire_test_data = pd.read_csv('datasets/wildfire_test.csv')
    wildfire_validation_data = pd.read_csv('datasets/wildfire_val.csv')

    print('Preprocessing data')
    print(f'wildfire_train_data shape: {wildfire_train_data.shape}')

    wildfire_drop_columns = ['Shape', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                             'NWCG_REPORTING_UNIT_ID',
                             'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME',
                             'LOCAL_FIRE_REPORT_ID',
                             'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME', 'ICS_209_PLUS_INCIDENT_JOIN_ID',
                             'ICS_209_PLUS_COMPLEX_JOIN_ID', 'MTBS_ID', 'MTBS_FIRE_NAME', 'COMPLEX_NAME',
                             'NWCG_CAUSE_CLASSIFICATION', 'NWCG_GENERAL_CAUSE',
                             'NWCG_CAUSE_AGE_CATEGORY', 'CONT_DATE', 'CONT_DOY', 'DISCOVERY_TIME',
                             'OWNER_DESCR', 'FOD_ID', 'FPA_ID', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'CONT_DATE',
                             'CONT_DOY', 'CONT_TIME',
                             'FIRE_SIZE_CLASS', 'STATE', 'COUNTY', 'FIPS_NAME']
    weather_drop_columns = ['WS10M_MIN', 'WS50M_MIN', 'WS50M', 'WS50M_RANGE', 'WS50M_MAX', 'T2MWET']
    wildfire_train_data = wildfire_train_data.drop(columns=wildfire_drop_columns, axis=1)
    wildfire_validation_data = wildfire_validation_data.drop(columns=wildfire_drop_columns, axis=1)
    wildfire_test_data = wildfire_test_data.drop(columns=wildfire_drop_columns, axis=1)
    weather_train_data = weather_train_data.drop(columns=weather_drop_columns, axis=1)
    weather_validation_data = weather_validation_data.drop(columns=weather_drop_columns, axis=1)
    weather_test_data = weather_test_data.drop(columns=weather_drop_columns, axis=1)
    wildfire_train_data['FIRE_YEAR'] = pd.to_numeric(wildfire_train_data['FIRE_YEAR'], errors='coerce')
    wildfire_test_data['FIRE_YEAR'] = pd.to_numeric(wildfire_test_data['FIRE_YEAR'], errors='coerce')
    wildfire_validation_data['FIRE_YEAR'] = pd.to_numeric(wildfire_validation_data['FIRE_YEAR'], errors='coerce')
    wildfire_train_data['FIRE_SIZE'] = pd.to_numeric(wildfire_train_data['FIRE_SIZE'], errors='coerce')
    wildfire_test_data['FIRE_SIZE'] = pd.to_numeric(wildfire_test_data['FIRE_SIZE'], errors='coerce')
    wildfire_validation_data['FIRE_SIZE'] = pd.to_numeric(wildfire_validation_data['FIRE_SIZE'], errors='coerce')
    wildfire_train_data['FIPS_CODE'] = pd.to_numeric(wildfire_train_data['FIPS_CODE'], errors='coerce')
    wildfire_test_data['FIPS_CODE'] = pd.to_numeric(wildfire_test_data['FIPS_CODE'], errors='coerce')
    wildfire_validation_data['FIPS_CODE'] = pd.to_numeric(wildfire_validation_data['FIPS_CODE'], errors='coerce')
    wildfire_train_data['FIRE_YEAR'] = wildfire_train_data['FIRE_YEAR'].fillna(0).astype(int)
    wildfire_test_data['FIRE_YEAR'] = wildfire_test_data['FIRE_YEAR'].fillna(0).astype(int)
    wildfire_validation_data['FIRE_YEAR'] = wildfire_validation_data['FIRE_YEAR'].fillna(0).astype(int)
    wildfire_train_data = wildfire_train_data[wildfire_train_data['FIRE_YEAR'] >= 2012]
    wildfire_validation_data = wildfire_validation_data[wildfire_validation_data['FIRE_YEAR'] >= 2012]
    wildfire_test_data = wildfire_test_data[wildfire_test_data['FIRE_YEAR'] >= 2012]
    weather_test_data = weather_test_data[weather_test_data['DATE'] >= '2015-01-01']
    weather_validation_data = weather_validation_data[weather_validation_data['DATE'] >= '2015-01-01']
    weather_train_data = weather_train_data[weather_train_data['DATE'] >= '2015-01-01']
    wildfire_train_data = wildfire_train_data.dropna(subset=['FIPS_CODE'])
    wildfire_test_data = wildfire_test_data.dropna(subset=['FIPS_CODE'])
    wildfire_validation_data = wildfire_validation_data.dropna(subset=['FIPS_CODE'])
    weather_train_data['FIPS_CODE'] = weather_train_data['FIPS_CODE'].astype(int)
    wildfire_train_data['FIPS_CODE'] = wildfire_train_data['FIPS_CODE'].astype(int)
    weather_test_data['FIPS_CODE'] = weather_test_data['FIPS_CODE'].astype(int)
    wildfire_test_data['FIPS_CODE'] = wildfire_test_data['FIPS_CODE'].astype(int)
    weather_validation_data['FIPS_CODE'] = weather_validation_data['FIPS_CODE'].astype(int)
    wildfire_validation_data['FIPS_CODE'] = wildfire_validation_data['FIPS_CODE'].astype(int)
    weather_train_data['FIPS_CODE'] = weather_train_data['FIPS_CODE'].astype(int)
    wildfire_train_data['FIPS_CODE'] = wildfire_train_data['FIPS_CODE'].astype(int)
    weather_test_data['FIPS_CODE'] = weather_test_data['FIPS_CODE'].astype(int)
    wildfire_test_data['FIPS_CODE'] = wildfire_test_data['FIPS_CODE'].astype(int)
    weather_validation_data['FIPS_CODE'] = weather_validation_data['FIPS_CODE'].astype(int)
    wildfire_validation_data['FIPS_CODE'] = wildfire_validation_data['FIPS_CODE'].astype(int)
    weather_train_data['DATE'] = pd.to_datetime(weather_train_data['DATE'])
    wildfire_train_data['DATE'] = pd.to_datetime(wildfire_train_data['DATE'])
    weather_test_data['DATE'] = pd.to_datetime(weather_test_data['DATE'])
    wildfire_test_data['DATE'] = pd.to_datetime(wildfire_test_data['DATE'])
    weather_validation_data['DATE'] = pd.to_datetime(weather_validation_data['DATE'])
    wildfire_validation_data['DATE'] = pd.to_datetime(wildfire_validation_data['DATE'])
    #print(f'wildfire_train_data shape: {wildfire_train_data.shape}')

    # Save the data to CSV files
    weather_train_data.to_csv('datasets/prep_saved_weather_train.csv', index=False)
    weather_test_data.to_csv('datasets/prep_saved_weather_test.csv', index=False)
    weather_validation_data.to_csv('datasets/prep_saved_weather_val.csv', index=False)
    wildfire_train_data.to_csv('datasets/prep_saved_wildfire_train.csv', index=False)
    wildfire_test_data.to_csv('datasets/prep_saved_wildfire_test.csv', index=False)
    wildfire_validation_data.to_csv('datasets/prep_saved_wildfire_val.csv', index=False)
    return weather_train_data, weather_test_data, weather_validation_data, wildfire_train_data, wildfire_test_data, wildfire_validation_data


def create_fips_wildfire_csv():
    df = pd.read_csv('datasets/state_and_county_fips_master.csv')

    df['WILDFIRE_OCCURRENCE'] = 0

    df.to_csv('datasets/FIPS_WILDFIRE', index=False)


def get_counties_for_state(state):
    print('Getting counties for state')
    try:
        df = pd.read_csv('datasets/state_and_county_fips_master.csv')
        try:
            counties = df[df['STATE'] == state]['COUNTY'].unique()
           # print(counties)

            return counties
        except KeyError:
            print('State not found')
    except FileNotFoundError:
       print('File not found')