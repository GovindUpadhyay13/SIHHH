"""Automated test suite for the three new CARLA 3D integration modules:
  - Module 1: Pothole Detection, Local Cost Map, and Alternate Route Event
  - Module 2: Crowd-Aware Slowdown with Density Bands & Persistent Zone Caching
  - Module 3: Weather-Aware Slowdown reading CARLA WeatherParameters
"""

import pytest
import time
from pathlib import Path

from src.models import VehicleState, Pothole, WeatherState, WeatherCondition
from src.perception.pothole_cost_map import PotholeCostMap, PotholeDetectorModule
from src.control.crowd_density_monitor import CrowdDensityMonitor, CrowdDensityBand
from src.control.weather_response import (
    WeatherResponseModule,
    CarlaWeatherParametersProxy,
    DrivingCautionLevel,
)


# ==========================================
# MODULE 1 TESTS: Pothole Cost Map
# ==========================================

def test_pothole_cost_map_decay_and_spatial_cost():
    """Verify that potholes raise local costs radially and decay over time."""
    cost_map = PotholeCostMap(grid_resolution_m=0.5, decay_half_life_s=2.0, influence_radius_m=2.0)
    now = time.time()

    # Add pothole at (10.0, 0.0)
    cost_map.add_pothole(x=10.0, y=0.0, size_m=0.8, confidence=0.9, now=now)

    # Cost directly at pothole center should be high
    center_cost = cost_map.get_cost_at(10.0, 0.0)
    assert center_cost > 5.0

    # Cost 1.0m away should be lower but positive
    offset_cost = cost_map.get_cost_at(11.0, 0.0)
    assert 0.0 < offset_cost < center_cost

    # Cost far away (5.0m) should be 0.0
    far_cost = cost_map.get_cost_at(15.0, 0.0)
    assert far_cost == 0.0

    # Test time decay
    cost_map.update_decay(now + 3.0)
    decayed_cost = cost_map.get_cost_at(10.0, 0.0)
    assert decayed_cost < center_cost


def test_pothole_density_triggers_reroute_event():
    """Verify that dense pothole clusters trigger the alternate route suggestion callback."""
    reroute_triggered = []

    def on_reroute(reason: str):
        reroute_triggered.append(reason)

    detector = PotholeDetectorModule(
        enabled=True,
        density_reroute_threshold=1.5,
        on_reroute_suggested=on_reroute,
    )

    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=10.0)
    sensor_data = {
        "detected_potholes": [
            Pothole(id="p1", x=10.0, y=0.0, road_segment_id="s1", severity=0.9, confidence=0.95),
            Pothole(id="p2", x=14.0, y=0.2, road_segment_id="s1", severity=0.85, confidence=0.95),
            Pothole(id="p3", x=18.0, y=-0.1, road_segment_id="s1", severity=0.95, confidence=0.95),
        ]
    }

    lookahead_path = [(float(x), 0.0) for x in range(5, 25, 2)]
    cost_map, speed_cap, event = detector.update(vehicle, sensor_data, lookahead_path=lookahead_path)

    assert event == "suggest_alternate_route"
    assert len(reroute_triggered) == 1
    assert speed_cap is not None
    assert speed_cap <= 20.0


def test_module1_toggle_flag():
    """Verify that disabling Module 1 returns None for speed cap and event."""
    detector = PotholeDetectorModule(enabled=False)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=10.0)
    sensor_data = {"detected_potholes": [Pothole(id="p1", x=5.0, y=0.0, road_segment_id="s1", severity=0.9, confidence=0.9)]}

    cost_map, speed_cap, event = detector.update(vehicle, sensor_data)
    assert speed_cap is None
    assert event is None


# ==========================================
# MODULE 2 TESTS: Crowd Density Monitor
# ==========================================

def test_crowd_density_bands_and_speed_capping(tmp_path):
    """Verify density computation, classification into bands, and speed cap emission."""
    cache_file = tmp_path / "crowd_cache.json"
    monitor = CrowdDensityMonitor(enabled=True, cache_file_path=cache_file)
    vehicle = VehicleState(x=50.0, y=0.0, heading=0.0, velocity=10.0)

    # 1. Low crowd (2 agents)
    perception_data_low = {"detected_agents": [{"x": 55.0, "y": 1.0}, {"x": 60.0, "y": -1.0}]}
    speed_cap, info = monitor.update(vehicle, perception_data_low, dt=0.1)
    assert info["density_band"] == CrowdDensityBand.LOW.value

    # 2. Dense crowd (18 agents directly in front)
    dense_agents = [{"x": 55.0 + i * 0.5, "y": (i % 3) - 1.0} for i in range(18)]
    perception_data_dense = {"detected_agents": dense_agents}

    # Simulate over several ticks to fill rolling window
    for _ in range(12):
        speed_cap, info = monitor.update(vehicle, perception_data_dense, dt=0.1)

    assert info["density_band"] in [CrowdDensityBand.HIGH.value, CrowdDensityBand.DENSE.value]
    assert speed_cap < 30.0


def test_crowd_persistent_zone_caching(tmp_path):
    """Verify that crowd-heavy locations are cached and trigger proactive approach slowing."""
    cache_file = tmp_path / "crowd_cache.json"
    monitor = CrowdDensityMonitor(enabled=True, cache_file_path=cache_file)
    vehicle = VehicleState(x=100.0, y=0.0, heading=0.0, velocity=10.0)

    # Ingest dense crowd at (100, 0)
    dense_agents = [{"x": 105.0 + i * 0.5, "y": 0.0} for i in range(20)]
    for _ in range(12):
        monitor.update(vehicle, {"detected_agents": dense_agents}, dt=0.1)

    # Cache file must have been written to disk
    assert cache_file.exists()
    assert len(monitor.known_zones) >= 1

    # Second run: vehicle approaches the cached zone with zero current agents
    vehicle_approach = VehicleState(x=85.0, y=0.0, heading=0.0, velocity=10.0)
    speed_cap, info = monitor.update(vehicle_approach, {"detected_agents": []}, dt=0.1)

    # Proactive slowing on approach due to cache hit
    assert info["cache_hit"] is True
    assert info["zone_id"] is not None


# ==========================================
# MODULE 3 TESTS: CARLA Weather Response
# ==========================================

def test_carla_weather_caution_levels():
    """Verify CARLA WeatherParameters proxy mapping to caution levels and speed caps."""
    weather_mod = WeatherResponseModule(enabled=True)

    # 1. Clear weather
    clear_carla = CarlaWeatherParametersProxy(precipitation=0.0, fog_density=0.0, precipitation_deposits=0.0)
    cap_clear, margin_clear, info_clear = weather_mod.update(clear_carla)
    assert info_clear["caution_level"] == DrivingCautionLevel.CLEAR.value
    assert cap_clear == 40.0
    assert margin_clear == 1.0

    # 2. Light rain (25% precipitation)
    light_carla = CarlaWeatherParametersProxy(precipitation=25.0, fog_density=10.0, precipitation_deposits=35.0)
    cap_light, margin_light, info_light = weather_mod.update(light_carla)
    assert info_light["caution_level"] == DrivingCautionLevel.LIGHT_RAIN.value
    assert cap_light == 32.0
    assert margin_light == 1.3

    # 3. Heavy rain (60% precipitation)
    heavy_carla = CarlaWeatherParametersProxy(precipitation=60.0, fog_density=40.0, precipitation_deposits=70.0)
    cap_heavy, margin_heavy, info_heavy = weather_mod.update(heavy_carla)
    assert info_heavy["caution_level"] == DrivingCautionLevel.HEAVY_RAIN_OR_FOG.value
    assert cap_heavy == 22.0
    assert margin_heavy == 1.7

    # 4. Severe conditions (85% precipitation, wet road)
    severe_carla = CarlaWeatherParametersProxy(precipitation=85.0, fog_density=75.0, precipitation_deposits=90.0)
    cap_severe, margin_severe, info_severe = weather_mod.update(severe_carla)
    assert info_severe["caution_level"] == DrivingCautionLevel.SEVERE.value
    assert cap_severe == 14.0
    assert margin_severe == 2.2


def test_uniform_module_interface_arbitration():
    """Verify uniform speed cap arbitration across all 3 modules."""
    m1 = PotholeDetectorModule(enabled=True)
    m2 = CrowdDensityMonitor(enabled=True)
    m3 = WeatherResponseModule(enabled=True)

    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=10.0)

    # M1 returns speed cap of 20 km/h
    _, cap_m1, _ = m1.update(
        vehicle,
        {"detected_potholes": [Pothole(id="p1", x=10.0, y=0.0, road_segment_id="s1", severity=0.8, confidence=0.9)]}
    )

    # M2 returns speed cap of 18 km/h (dense crowd)
    dense_agents = [{"x": 10.0 + i, "y": 0.0} for i in range(16)]
    for _ in range(10):
        cap_m2, _ = m2.update(vehicle, {"detected_agents": dense_agents})

    # M3 returns speed cap of 22 km/h (heavy rain)
    heavy_carla = CarlaWeatherParametersProxy(precipitation=60.0)
    cap_m3, _, _ = m3.update(heavy_carla)

    # Uniform arbitration: minimum of all active speed caps
    caps = [cap for cap in [40.0, cap_m1, cap_m2, cap_m3] if cap is not None]
    effective_cap = min(caps)

    # Most restrictive is M2 crowd speed cap (~18 km/h or less)
    assert effective_cap <= 20.0
