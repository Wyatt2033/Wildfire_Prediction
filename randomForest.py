import datetime
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.model_selection import train_test_split


def split_data(merged_data):
    merged_data['WILDFIRE_OCCURRENCE'] = merged_data['FIRE_SIZE'] > merged_data['FIRE_SIZE'].quantile(0.75).astype(int)
    merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
    merged_data['MONTH'] = merged_data['DATE'].dt.month
    merged_data['DAY_OF_YEAR'] = merged_data['DATE'].dt.dayofyear

    features = [
        'T2M',  # Temperature at 2 Meters
        'T2M_MAX',  # Maximum Temperature at 2 Meters
        'T2M_MIN',  # Minimum Temperature at 2 Meters
        'T2M_RANGE',  # Temperature Range at 2 Meters
        'T2MDEW',  # Dew/Frost Point at 2 Meters
        'WS10M',  # Wind Speed at 10 Meters
        'WS10M_MAX',  # Maximum Wind Speed at 10 Meters
        'WS10M_RANGE',  # Wind Speed Range at 10 Meters
        'QV2M',  # Specific Humidity at 2 Meters
        'PRECTOT',  # Total Precipitation
        'PS',  # Surface Pressure
        'MONTH',  # Month extracted from DATE
        'DAY_OF_YEAR'  # Day of the year extracted from DATE
    ]
    x = merged_data[features]
    y = merged_data['WILDFIRE_OCCURRENCE']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    return model


def save_model(model):
    pickle.dumps(model)
    with open('./models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def predict_wildfire_risk(weather_data_averages):
    with open('./models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    now = datetime.datetime.now()
    month = now.month
    day_of_year = now.timetuple().tm_yday
    # These are the features used to train the model
    training_features = ['T2M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'T2MDEW', 'WS10M', 'WS10M_MAX', 'WS10M_RANGE',
                         'QV2M', 'PRECTOT', 'PS', 'MONTH', 'DAY_OF_YEAR', 'DAY', 'FIPS_CODE', 'FIRE_SIZE', 'FIRE_YEAR',
                         'OBJECTID']

    data = {feature: [0] for feature in training_features}
    for feature in weather_data_averages.columns:
        data[feature] = [weather_data_averages[feature].values[0]]

    data = pd.DataFrame(data)

    data = pd.DataFrame({
        'T2M': [weather_data_averages['temperature_2m_max']+weather_data_averages['temperature_2m_min']/2],
        'T2M_MAX': [weather_data_averages['temperature_2m_max']],
        'T2M_MIN': [weather_data_averages['temperature_2m_min']],
        'T2M_RANGE': [weather_data_averages['temperature_2m_max'] - weather_data_averages['temperature_2m_min']],
        'T2MDEW': [weather_data_averages['dew_point_2m']],
        'WS10M': [weather_data_averages['wind_speed_10m_max']],
        'WS10M_MAX': [weather_data_averages['wind_speed_10m_max']],
        'WS10M_RANGE': [weather_data_averages['wind_gusts_10m_max'] - weather_data_averages['wind_speed_10m_max']],
        'QV2M': [weather_data_averages['relative_humidity_2m']],
        'PRECTOT': [weather_data_averages['precipitation_sum']],
        'PS': [weather_data_averages['surface_pressure']],
        'MONTH': [month],
        'DAY_OF_YEAR': [day_of_year]
    })

    prediction = model.predict(data)
    probability = model.predict_proba(data)
    wildfire_probability = probability[0][1]
    return prediction, wildfire_probability