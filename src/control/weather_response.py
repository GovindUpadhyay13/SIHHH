"""Module 3: CARLA Weather-Aware Slowdown & Driving Caution Adaptation.

Reads CARLA's weather state (rain intensity, fog density, wetness/precipitation deposits, sun altitude),
maps weather severity to driving-caution levels (CLEAR, LIGHT_RAIN, HEAVY_RAIN_OR_FOG, SEVERE),
and emits a target-speed-cap signal and safety margin expansion factor using the uniform controller interface.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from src.models import WeatherState

logger = logging.getLogger("WeatherResponseModule")


class DrivingCautionLevel(str, Enum):
    CLEAR = "CLEAR"
    LIGHT_RAIN = "LIGHT_RAIN"
    HEAVY_RAIN_OR_FOG = "HEAVY_RAIN_OR_FOG"
    SEVERE = "SEVERE"


@dataclass
class CarlaWeatherParametersProxy:
    """
    Drop-in representation of carla.WeatherParameters for environments where
    CARLA 3D server is connected or standalone simulation mode is running.
    """
    precipitation: float = 0.0           # 0.0 to 100.0 (rain intensity percentage)
    precipitation_deposits: float = 0.0  # 0.0 to 100.0 (road wetness/puddles percentage)
    fog_density: float = 0.0             # 0.0 to 100.0 (fog density percentage)
    sun_altitude_angle: float = 45.0     # -90.0 (night) to 90.0 (midday)
    wind_intensity: float = 0.0          # 0.0 to 100.0


@dataclass
class WeatherLogRecord:
    timestamp: float
    caution_level: str
    precipitation: float
    fog_density: float
    wetness: float
    speed_cap_kmh: float
    safety_margin_multiplier: float


class WeatherResponseModule:
    """
    Module 3: Pluggable CARLA weather response controller.
    Consumes ground-truth weather state, computes caution level,
    and returns a target-speed-cap signal and following-distance multiplier.
    """

    def __init__(
        self,
        enabled: bool = True,
        # Tunable thresholds
        light_rain_threshold: float = 15.0,        # precipitation %
        heavy_rain_threshold: float = 50.0,        # precipitation %
        fog_caution_threshold: float = 35.0,       # fog %
        severe_wetness_threshold: float = 75.0,    # wetness deposits %
        # Speed caps in km/h
        base_speed_kmh: float = 40.0,
        light_rain_cap_kmh: float = 32.0,
        heavy_rain_cap_kmh: float = 22.0,
        severe_cap_kmh: float = 14.0,
    ):
        self.enabled = enabled
        self.light_rain_threshold = light_rain_threshold
        self.heavy_rain_threshold = heavy_rain_threshold
        self.fog_caution_threshold = fog_caution_threshold
        self.severe_wetness_threshold = severe_wetness_threshold

        self.base_speed_kmh = base_speed_kmh
        self.light_rain_cap_kmh = light_rain_cap_kmh
        self.heavy_rain_cap_kmh = heavy_rain_cap_kmh
        self.severe_cap_kmh = severe_cap_kmh

        self.logs: list[WeatherLogRecord] = []

    def update(
        self,
        carla_weather_or_state: Any,  # carla.WeatherParameters instance or CarlaWeatherParametersProxy or WeatherState
        dt: float = 0.1,
    ) -> Tuple[Optional[float], float, Dict[str, Any]]:
        """
        Uniform integration tick:
        Returns:
          - speed_cap_kmh: Maximum allowed speed under current weather caution level.
          - safety_margin_multiplier: Headway / following distance multiplier (e.g. 1.0x to 2.2x).
          - telemetry_info: Dict containing caution level and weather indicators.
        """
        if not self.enabled:
            return None, 1.0, {"enabled": False}

        now = time.time()

        # 1. Normalize weather inputs from CARLA WeatherParameters or simulation state
        precip = 0.0
        wetness = 0.0
        fog = 0.0
        sun_alt = 45.0

        if hasattr(carla_weather_or_state, "precipitation"):
            precip = float(carla_weather_or_state.precipitation)
            wetness = float(getattr(carla_weather_or_state, "precipitation_deposits", 0.0))
            fog = float(getattr(carla_weather_or_state, "fog_density", 0.0))
            sun_alt = float(getattr(carla_weather_or_state, "sun_altitude_angle", 45.0))
        elif isinstance(carla_weather_or_state, WeatherState):
            # Fallback from existing simulation weather state
            precip = carla_weather_or_state.rain_intensity * 100.0
            wetness = (1.0 - (carla_weather_or_state.friction_coefficient / 0.85)) * 100.0
            fog = max(0.0, (1.0 - carla_weather_or_state.visibility_meters / 250.0) * 100.0)

        # 2. Map to Driving Caution Levels
        is_severe = (
            precip >= 75.0 or
            fog >= 70.0 or
            (precip >= self.heavy_rain_threshold and wetness >= self.severe_wetness_threshold)
        )
        is_heavy = (
            precip >= self.heavy_rain_threshold or
            fog >= self.fog_caution_threshold
        )
        is_light = (
            precip >= self.light_rain_threshold or
            wetness >= 30.0
        )

        if is_severe:
            caution = DrivingCautionLevel.SEVERE
            speed_cap = self.severe_cap_kmh
            margin_mult = 2.2
        elif is_heavy:
            caution = DrivingCautionLevel.HEAVY_RAIN_OR_FOG
            speed_cap = self.heavy_rain_cap_kmh
            margin_mult = 1.7
        elif is_light:
            caution = DrivingCautionLevel.LIGHT_RAIN
            speed_cap = self.light_rain_cap_kmh
            margin_mult = 1.3
        else:
            caution = DrivingCautionLevel.CLEAR
            speed_cap = self.base_speed_kmh
            margin_mult = 1.0

        # Log event
        self.logs.append(
            WeatherLogRecord(
                timestamp=now,
                caution_level=caution.value,
                precipitation=round(precip, 1),
                fog_density=round(fog, 1),
                wetness=round(wetness, 1),
                speed_cap_kmh=speed_cap,
                safety_margin_multiplier=margin_mult,
            )
        )

        return speed_cap, margin_mult, {
            "caution_level": caution.value,
            "precipitation": round(precip, 1),
            "fog_density": round(fog, 1),
            "wetness": round(wetness, 1),
            "speed_cap_kmh": speed_cap,
            "safety_margin_multiplier": margin_mult,
        }
