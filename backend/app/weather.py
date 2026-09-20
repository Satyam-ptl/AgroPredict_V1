"""
OpenWeather integration: turns a city name or lat/long into the three weather features
the model needs (temperature_C, humidity_percent, rainfall_mm), so the caller doesn't
have to type them in by hand.

Requires OPENWEATHER_API_KEY to be set as an environment variable.
"""
import os
from typing import Optional

import httpx
from fastapi import HTTPException

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherReading:
    def __init__(self, temperature_C: float, humidity_percent: float, rainfall_mm: float, source: str):
        self.temperature_C = temperature_C
        self.humidity_percent = humidity_percent
        self.rainfall_mm = rainfall_mm
        self.source = source


def _get_api_key() -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENWEATHER_API_KEY is not set on the server. "
                   "Set it as an environment variable, or use /predict (manual weather input) instead.",
        )
    return api_key


def _parse_response(data: dict, source: str) -> WeatherReading:
    try:
        temperature_C = data["main"]["temp"]
        humidity_percent = data["main"]["humidity"]
    except KeyError as e:
        raise HTTPException(status_code=502, detail=f"Unexpected OpenWeather response shape: missing {e}")

    # OpenWeather only reports 'rain' when it's actually raining -- 0mm is the correct
    # default otherwise, not a missing-data error. Prefers the 1h figure; falls back to 3h.
    rain_block = data.get("rain", {})
    rainfall_mm = rain_block.get("1h", rain_block.get("3h", 0.0))

    return WeatherReading(
        temperature_C=float(temperature_C),
        humidity_percent=float(humidity_percent),
        rainfall_mm=float(rainfall_mm),
        source=source,
    )


async def get_weather_by_city(city: str) -> WeatherReading:
    api_key = _get_api_key()
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPENWEATHER_BASE_URL, params=params)
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"City not found by OpenWeather: {city!r}")
    resp.raise_for_status()
    return _parse_response(resp.json(), source=f"openweather:{city}")


async def get_weather_by_coords(latitude: float, longitude: float) -> WeatherReading:
    api_key = _get_api_key()
    params = {"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPENWEATHER_BASE_URL, params=params)
    resp.raise_for_status()
    return _parse_response(resp.json(), source=f"openweather:{latitude},{longitude}")


async def resolve_weather(city: Optional[str], latitude: Optional[float], longitude: Optional[float]) -> WeatherReading:
    if city:
        return await get_weather_by_city(city)
    if latitude is not None and longitude is not None:
        return await get_weather_by_coords(latitude, longitude)
    raise HTTPException(
        status_code=422,
        detail="Provide either 'city', or both 'latitude' and 'longitude', for auto weather lookup.",
    )
