"""Unit tests for SP2: Crowd-Aware Adaptive Speed."""

import pytest
from src.models import VehicleState, RoadSegment, CrowdObservation, WeatherState, WeatherCondition
from src.control.speed_controller import AdaptiveSpeedController
from src.perception.crowd_provider import ConfiguredCrowdProvider


@pytest.fixture
def base_road_segment():
    return RoadSegment(
        id="seg_test",
        start_node="A",
        end_node="B",
        length_meters=100.0,
        speed_limit_kmh=40.0,
    )


@pytest.fixture
def clear_weather():
    return WeatherState(
        condition=WeatherCondition.CLEAR,
        rain_intensity=0.0,
        visibility_meters=200.0,
        friction_coefficient=0.85,
    )


def test_low_crowd_produces_normal_speed(base_road_segment, clear_weather):
    """Verify that absent/low crowd density maintains full base cruise speed."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    crowd_obs = CrowdObservation(
        road_segment_id="seg_test",
        current_density=0.05,
        distance_to_crowd_meters=50.0,
        is_active=False,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=base_road_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.05,
        crowd_observation=crowd_obs,
        weather_state=clear_weather,
    )

    assert result.target_speed_kmh == 40.0
    assert result.limiting_factor in ["base_cruise", "road_speed_limit"]


def test_high_historical_crowd_reduces_speed(base_road_segment, clear_weather):
    """Verify that a road segment with high historical crowd density reduces target speed."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    # Segment historically known as dense bazaar, vehicle approaching at 20m
    crowd_obs = CrowdObservation(
        road_segment_id="seg_test",
        current_density=0.10,
        distance_to_crowd_meters=20.0,
        is_active=True,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=base_road_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.85,
        crowd_observation=crowd_obs,
        weather_state=clear_weather,
    )

    assert result.crowd_safety_speed_kmh < 35.0
    assert result.target_speed_kmh < 40.0
    assert any("crowd awareness" in r.lower() for r in result.reasons)


def test_high_current_crowd_reduces_speed_further(base_road_segment, clear_weather):
    """Verify that an active dense pedestrian cluster produces deeper speed reduction."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    # Real-time observation of dense crowd close ahead (8m)
    crowd_obs = CrowdObservation(
        road_segment_id="seg_test",
        current_density=0.95,
        distance_to_crowd_meters=8.0,
        is_active=True,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=base_road_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.85,
        crowd_observation=crowd_obs,
        weather_state=clear_weather,
    )

    # Deep speed reduction down towards crowd minimum safety limit (~12-15 km/h)
    assert result.crowd_safety_speed_kmh <= 18.0
    assert result.target_speed_kmh < 30.0
    assert result.limiting_factor == "crowd_density"


def test_crowd_speed_never_exceeds_road_limit(clear_weather):
    """Verify that speed remains strictly bounded by road limit even with zero crowd."""
    tight_segment = RoadSegment(
        id="seg_school",
        start_node="A",
        end_node="B",
        length_meters=50.0,
        speed_limit_kmh=25.0,  # Strict school zone speed limit
    )
    controller = AdaptiveSpeedController(base_speed_kmh=50.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=50.0 / 3.6)

    crowd_obs = CrowdObservation(
        road_segment_id="seg_school",
        current_density=0.0,
        distance_to_crowd_meters=100.0,
        is_active=False,
    )

    result = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=tight_segment,
        pothole_safety_speed_kmh=50.0,
        historical_crowd_score=0.0,
        crowd_observation=crowd_obs,
        weather_state=clear_weather,
    )

    assert result.target_speed_kmh <= 25.0


def test_speed_recovers_after_leaving_crowded_region(base_road_segment, clear_weather):
    """Verify that vehicle target speed smoothly recovers after leaving crowded segment."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=15.0 / 3.6)

    # 1. Inside crowd zone
    crowd_active = CrowdObservation(
        road_segment_id="seg_test",
        current_density=0.90,
        distance_to_crowd_meters=5.0,
        is_active=True,
    )
    res_in_crowd = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=base_road_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.80,
        crowd_observation=crowd_active,
        weather_state=clear_weather,
        dt=0.1,
    )
    assert res_in_crowd.target_speed_kmh <= 20.0

    # 2. Exited crowd zone
    crowd_cleared = CrowdObservation(
        road_segment_id="seg_test",
        current_density=0.0,
        distance_to_crowd_meters=100.0,
        is_active=False,
    )
    # Simulate over 4 seconds of recovery
    for _ in range(40):
        res_after = controller.compute_target_speed(
            vehicle=vehicle,
            current_segment=base_road_segment,
            pothole_safety_speed_kmh=40.0,
            historical_crowd_score=0.05,
            crowd_observation=crowd_cleared,
            weather_state=clear_weather,
            dt=0.1,
        )

    # Target speed should have recovered to normal base cruise speed
    assert res_after.target_speed_kmh == 40.0
