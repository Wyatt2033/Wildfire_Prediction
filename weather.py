import os

import openmeteo_requests
import requests_cache
import pandas as pd
import json
from retry_requests import retry


def get_cached_weather_data(lat, long):
    cache_file_path = 'weather_cache.json'
    cache_key = f'{lat}_{long}'
    cache = {}

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            for line in f:
                try:
                    cache.update(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {line}. Error: {e}")
                    continue

    if cache_key in cache:
        weather_data = pd.DataFrame(cache[cache_key])
    else:
        weather_data = get_weather_data(lat, long)
        cache[cache_key] = weather_data.to_dict(orient='records')
        with open(cache_file_path, 'w') as f:
            json.dump(cache, f)
    weather_data = pd.DataFrame(weather_data)
    return weather_data


def get_weather_data(lat, long):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": long,
        "current": ["relative_humidity_2m", "surface_pressure"],
        "hourly": "dew_point_2m",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max",
                  "wind_gusts_10m_max"],
        "past_days": 7
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()

    current_relative_humidity_2m = current.Variables(0).Value()
    current_surface_pressure = current.Variables(1).Value()

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_dew_point_2m = hourly.Variables(0).ValuesAsNumpy()[:14]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(4).ValuesAsNumpy()
    current_relative_humidity_2m = current_relative_humidity_2m
    current_surface_pressure = current_surface_pressure
    hourly_dew_point_2m = hourly_dew_point_2m

    daily_data = {"temperature_2m_max": daily_temperature_2m_max, "temperature_2m_min": daily_temperature_2m_min,
                  "precipitation_sum": daily_precipitation_sum, "wind_speed_10m_max": daily_wind_speed_10m_max,
                  "wind_gusts_10m_max": daily_wind_gusts_10m_max, "surface_pressure": current_surface_pressure,
                  "relative_humidity_2m": current_relative_humidity_2m, "dew_point_2m": hourly_dew_point_2m}

    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe
