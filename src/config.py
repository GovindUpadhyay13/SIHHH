"""Centralized configuration and parameters for the autonomous driving simulation."""

from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase_m: float = 2.7
    width_m: float = 1.8
    length_m: float = 4.2
    max_steer_rad: float = 0.60  # ~34.4 degrees
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.5
    emergency_decel_mps2: float = 6.0


@dataclass(frozen=True)
class SpeedConfig:
    min_safe_speed_kmh: float = 10.0
    base_cruise_speed_kmh: float = 40.0
    max_vehicle_speed_kmh: float = 50.0
    comfort_decel_rate_kmh_per_s: float = 8.0  # Gradual speed reduction


@dataclass(frozen=True)
class PerceptionConfig:
    pothole_detection_range_m: float = 25.0
    pothole_fov_deg: float = 90.0
    min_detection_confidence: float = 0.50


@dataclass(frozen=True)
class RoadMemoryConfig:
    db_path: Path = DATA_DIR / "road_memory.db"
    spatial_dedup_radius_m: float = 2.0
    route_replan_risk_threshold: float = 1.2
    pothole_risk_weight: float = 1.0


@dataclass(frozen=True)
class CrowdConfig:
    warning_distance_m: float = 35.0
    critical_distance_m: float = 8.0
    historical_weight: float = 0.40
    current_weight: float = 0.60
    min_speed_in_dense_crowd_kmh: float = 12.0


@dataclass(frozen=True)
class WeatherConfig:
    light_rain_speed_factor: float = 0.80  # 20% reduction
    heavy_rain_speed_factor: float = 0.50  # 50% reduction
    light_rain_friction: float = 0.65
    heavy_rain_friction: float = 0.40
    clear_friction: float = 0.85
    clear_headway_s: float = 2.0
    light_rain_headway_s: float = 3.0
    heavy_rain_headway_s: float = 4.5
    api_timeout_seconds: float = 3.0


VEHICLE_CONFIG = VehicleConfig()
SPEED_CONFIG = SpeedConfig()
PERCEPTION_CONFIG = PerceptionConfig()
ROAD_MEMORY_CONFIG = RoadMemoryConfig()
CROWD_CONFIG = CrowdConfig()
WEATHER_CONFIG = WeatherConfig()
