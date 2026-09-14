"""Weather provider interface, deterministic simulation provider, and live weather API provider."""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import time
import logging
from typing import Optional, List, Tuple
import requests

from src.models import WeatherState, WeatherCondition, WeatherProviderMode
from src.config import WEATHER_CONFIG

logger = logging.getLogger(__name__)


class WeatherProvider(ABC):
    """Abstract interface for weather state providers."""

    @abstractmethod
    def get_weather(self, sim_time_seconds: float = 0.0) -> WeatherState:
        """Fetch current weather state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> WeatherProviderMode:
        raise NotImplementedError


class DeterministicMockWeatherProvider(WeatherProvider):
    """
    Deterministic mock weather provider for offline demonstration and reproducible testing.
    Transitions deterministically across:
    0s - 15s: CLEAR
    15s - 35s: LIGHT_RAIN
    35s - 55s: HEAVY_RAIN
    55s+: CLEAR
    """

    def __init__(
        self,
        schedule: Optional[List[Tuple[float, WeatherCondition]]] = None,
    ):
        # Default deterministic timeline (start_time_s, condition)
        self.schedule: List[Tuple[float, WeatherCondition]] = schedule or [
            (0.0, WeatherCondition.CLEAR),
            (15.0, WeatherCondition.LIGHT_RAIN),
            (35.0, WeatherCondition.HEAVY_RAIN),
            (55.0, WeatherCondition.CLEAR),
        ]
        self._manual_override: Optional[WeatherCondition] = None

    @property
    def mode(self) -> WeatherProviderMode:
        return WeatherProviderMode.MOCK

    def set_condition(self, condition: WeatherCondition) -> None:
        """Allows manual testing of specific weather states."""
        self._manual_override = condition

    def get_weather(self, sim_time_seconds: float = 0.0) -> WeatherState:
        """Returns deterministic weather state based on elapsed simulation time or override."""
        cond = WeatherCondition.CLEAR

        if self._manual_override is not None:
            cond = self._manual_override
        else:
            # Find the active schedule phase
            for start_t, condition in self.schedule:
                if sim_time_seconds >= start_t:
                    cond = condition

        if cond == WeatherCondition.CLEAR:
            return WeatherState(
                condition=WeatherCondition.CLEAR,
                rain_intensity=0.0,
                visibility_meters=250.0,
                friction_coefficient=WEATHER_CONFIG.clear_friction,
                provider_mode=WeatherProviderMode.MOCK,
                description="Simulated clear dry conditions",
                timestamp=time.time(),
            )
        elif cond == WeatherCondition.LIGHT_RAIN:
            return WeatherState(
                condition=WeatherCondition.LIGHT_RAIN,
                rain_intensity=0.35,
                visibility_meters=120.0,
                friction_coefficient=WEATHER_CONFIG.light_rain_friction,
                provider_mode=WeatherProviderMode.MOCK,
                description="Simulated light precipitation, reduced friction",
                timestamp=time.time(),
            )
        else:  # HEAVY_RAIN
            return WeatherState(
                condition=WeatherCondition.HEAVY_RAIN,
                rain_intensity=0.85,
                visibility_meters=45.0,
                friction_coefficient=WEATHER_CONFIG.heavy_rain_friction,
                provider_mode=WeatherProviderMode.MOCK,
                description="Simulated torrential rain, hazardous wet surface",
                timestamp=time.time(),
            )


class OpenWeatherMapProvider(WeatherProvider):
    """
    Live Weather API client for OpenWeatherMap.
    Reads API key from environment variable OPENWEATHER_API_KEY.
    If key is missing, network is unavailable, or response is invalid,
    it logs an explicit diagnostic and falls back gracefully to the mock provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        city: str = "Bengaluru",
        timeout: float = WEATHER_CONFIG.api_timeout_seconds,
        fallback_provider: Optional[WeatherProvider] = None,
    ):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.city = city
        self.timeout = timeout
        self.fallback_provider = fallback_provider or DeterministicMockWeatherProvider()
        self._last_successful_state: Optional[WeatherState] = None

    @property
    def mode(self) -> WeatherProviderMode:
        return WeatherProviderMode.LIVE if self.api_key else WeatherProviderMode.MOCK

    def get_weather(self, sim_time_seconds: float = 0.0) -> WeatherState:
        """Fetches live weather, falling back gracefully to mock if unavailable."""
        if not self.api_key:
            logger.info("OPENWEATHER_API_KEY not configured. Running in explicit MOCK mode.")
            return self.fallback_provider.get_weather(sim_time_seconds)

        url = f"https://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                weather_main = data.get("weather", [{}])[0].get("main", "Clear").lower()
                vis_m = float(data.get("visibility", 10000)) / 40.0  # Scale down to urban perception scale
                temp_c = float(data.get("main", {}).get("temp", 28.0))
                desc = data.get("weather", [{}])[0].get("description", "Live weather")

                if "rain" in weather_main or "drizzle" in weather_main:
                    rain_vol = data.get("rain", {}).get("1h", 1.5)
                    cond = WeatherCondition.HEAVY_RAIN if rain_vol > 5.0 else WeatherCondition.LIGHT_RAIN
                    rain_intensity = min(1.0, rain_vol / 10.0)
                    friction = WEATHER_CONFIG.heavy_rain_friction if cond == WeatherCondition.HEAVY_RAIN else WEATHER_CONFIG.light_rain_friction
                else:
                    cond = WeatherCondition.CLEAR
                    rain_intensity = 0.0
                    friction = WEATHER_CONFIG.clear_friction

                state = WeatherState(
                    condition=cond,
                    rain_intensity=rain_intensity,
                    visibility_meters=max(30.0, min(300.0, vis_m)),
                    friction_coefficient=friction,
                    provider_mode=WeatherProviderMode.LIVE,
                    temperature_celsius=temp_c,
                    description=f"Live OpenWeatherMap ({self.city}): {desc}",
                    timestamp=time.time(),
                )
                self._last_successful_state = state
                return state
            else:
                logger.warning(
                    f"OpenWeatherMap API error code {response.status_code}. Falling back to deterministic MOCK mode."
                )
                return self.fallback_provider.get_weather(sim_time_seconds)

        except Exception as ex:
            logger.warning(
                f"OpenWeatherMap network request failed ({ex}). Falling back to deterministic MOCK mode."
            )
            return self.fallback_provider.get_weather(sim_time_seconds)
