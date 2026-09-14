"""End-to-End Integration Tests for all three Smart Features operating concurrently."""

import pytest
from pathlib import Path

from src.models import (
    VehicleState,
    RoadSegment,
    Waypoint,
    Pothole,
    CrowdObservation,
    WeatherState,
    WeatherCondition,
    WeatherProviderMode,
)
from src.control.speed_controller import AdaptiveSpeedController
from src.memory.road_memory import SQLiteRoadMemoryStore
from src.perception.crowd_provider import ConfiguredCrowdProvider
from src.perception.weather_provider import DeterministicMockWeatherProvider
from src.simulation.simulation import (
    SimulationEnvironment,
    create_default_indian_road_network,
)
from src.config import SPEED_CONFIG


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "integration_road_memory.db"


def test_all_three_features_operate_simultaneously():
    """Verify that speed arbitration selects the most restrictive constraint when multiple hazards exist."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    segment = RoadSegment(
        id="seg_busy",
        start_node="A",
        end_node="B",
        length_meters=100.0,
        speed_limit_kmh=50.0,
    )

    # Condition 1: Pothole safety speed = 25 km/h
    pothole_safe_speed = 25.0

    # Condition 2: High crowd safety speed = 16 km/h
    crowd_obs = CrowdObservation(
        road_segment_id="seg_busy",
        current_density=0.85,
        distance_to_crowd_meters=10.0,
        is_active=True,
    )

    # Condition 3: Heavy rain safety speed = 20 km/h
    heavy_rain = WeatherState(
        condition=WeatherCondition.HEAVY_RAIN,
        rain_intensity=0.85,
        visibility_meters=50.0,
        friction_coefficient=0.40,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=segment,
        pothole_safety_speed_kmh=pothole_safe_speed,
        historical_crowd_score=0.80,
        crowd_observation=crowd_obs,
        weather_state=heavy_rain,
    )

    # Most restrictive active condition here is crowd (~16.0 km/h)
    assert result.target_speed_kmh <= 20.0
    assert result.limiting_factor == "crowd_density"
    assert len(result.reasons) >= 2


def test_minimum_speed_rule_is_strictly_respected():
    """Verify target speed never drops below configured minimum safety floor (10 km/h)."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0, min_safe_speed_kmh=10.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=10.0 / 3.6)

    segment = RoadSegment(id="s", start_node="A", end_node="B", length_meters=50.0, speed_limit_kmh=40.0)

    # Force all conditions to extreme minimums
    extreme_crowd = CrowdObservation(
        road_segment_id="s",
        current_density=1.0,
        distance_to_crowd_meters=1.0,
        is_active=True,
    )
    extreme_rain = WeatherState(
        condition=WeatherCondition.HEAVY_RAIN,
        rain_intensity=1.0,
        visibility_meters=10.0,
        friction_coefficient=0.20,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=segment,
        pothole_safety_speed_kmh=5.0,  # Below floor
        historical_crowd_score=1.0,
        crowd_observation=extreme_crowd,
        weather_state=extreme_rain,
    )

    assert result.target_speed_kmh >= 10.0
    assert result.target_speed_kmh >= SPEED_CONFIG.min_safe_speed_kmh


def test_full_offline_simulation_pipeline_without_external_api_or_ml_model(temp_db_path):
    """Verify that full end-to-end simulation runs completely offline without any external APIs or models."""
    network, potholes = create_default_indian_road_network()
    store = SQLiteRoadMemoryStore(db_path=temp_db_path)
    store.clear()

    crowd_provider = ConfiguredCrowdProvider()
    crowd_provider.add_dynamic_cluster("seg_main_arterial", s_center=60.0, density=0.80)

    weather_provider = DeterministicMockWeatherProvider()

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=potholes,
        road_memory=store,
        crowd_provider=crowd_provider,
        weather_provider=weather_provider,
        verbose_logging=False,
    )

    # Trip 1
    t1_summary = env.run_trip(trip_id="integration_trip_1", consider_road_memory=False, max_duration_s=25.0)
    assert t1_summary.potholes_detected >= 1
    assert t1_summary.potholes_stored_new >= 1
    assert t1_summary.is_alternate_route is False

    # Trip 2 (Memory informed)
    t2_summary = env.run_trip(trip_id="integration_trip_2", consider_road_memory=True, max_duration_s=25.0)
    assert t2_summary.is_alternate_route is True
    assert t2_summary.selected_route_segments == ["seg_bypass_leg1", "seg_bypass_leg2"]

    # Verify log explainability
    step_records = env.logger.step_records
    assert len(step_records) > 0
    sample_log = step_records[min(20, len(step_records) - 1)]
    assert sample_log.limiting_factor is not None
    assert len(sample_log.explanations) > 0
