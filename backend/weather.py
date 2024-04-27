import logging
import os
import openmeteo_requests
import requests_cache
import pandas as pd
import json
from retry_requests import retry


CACHE_FILE_PATH = 'cache/weather_cache/weather_cache.json'
CACHE_SESSION_PATH = '.cache'
API_URL = "https://api.open-meteo.com/v1/forecast"

def get_cached_weather_data(lat, long):
    """
    Fetches weather data for a given latitude and longitude from cache or API.
    """
    cache_key = f'{lat}_{long}'
    cache = {}

    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'r') as f:
            for line in f:
                try:
                    cache.update(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from line: {line}. Error: {e}")
                    continue

    if cache_key in cache:
        weather_data = pd.DataFrame(cache[cache_key])
    else:
        weather_data = get_weather_data(lat, long)
        cache[cache_key] = weather_data.to_dict(orient='records')
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache, f)
    weather_data = pd.DataFrame(weather_data)
    return weather_data


def get_weather_data(lat, long):
    """
    Fetches weather data for a given latitude and longitude from the API.
    """
    cache_session = requests_cache.CachedSession(CACHE_SESSION_PATH, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": long,
        "current": ["relative_humidity_2m", "surface_pressure"],
        "hourly": "dew_point_2m",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max",
                  "wind_gusts_10m_max"],
        "past_days": 7
    }
    responses = openmeteo.weather_api(API_URL, params=params)

    response = responses[0]

    current = response.Current()

    current_relative_humidity_2m = current.Variables(0).Value()
    current_surface_pressure = current.Variables(1).Value()

    hourly = response.Hourly()
    hourly_dew_point_2m = hourly.Variables(0).ValuesAsNumpy()[:14]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(4).ValuesAsNumpy()

    daily_data = {"temperature_2m_max": daily_temperature_2m_max, "temperature_2m_min": daily_temperature_2m_min,
                  "precipitation_sum": daily_precipitation_sum, "wind_speed_10m_max": daily_wind_speed_10m_max,
                  "wind_gusts_10m_max": daily_wind_gusts_10m_max, "surface_pressure": current_surface_pressure,
                  "relative_humidity_2m": current_relative_humidity_2m, "dew_point_2m": hourly_dew_point_2m}

    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe