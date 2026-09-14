"""Unit tests for SP3: Weather-Aware Adaptive Speed & Resilient API / Mock Modes."""

import pytest
from unittest.mock import patch, MagicMock

from src.models import (
    VehicleState,
    RoadSegment,
    CrowdObservation,
    WeatherState,
    WeatherCondition,
    WeatherProviderMode,
)
from src.control.speed_controller import AdaptiveSpeedController
from src.perception.weather_provider import (
    DeterministicMockWeatherProvider,
    OpenWeatherMapProvider,
)
from src.config import WEATHER_CONFIG


@pytest.fixture
def test_segment():
    return RoadSegment(
        id="s_test",
        start_node="A",
        end_node="B",
        length_meters=100.0,
        speed_limit_kmh=40.0,
    )


@pytest.fixture
def neutral_crowd():
    return CrowdObservation(
        road_segment_id="s_test",
        current_density=0.0,
        distance_to_crowd_meters=100.0,
        is_active=False,
    )


def test_clear_weather_produces_normal_speed(test_segment, neutral_crowd):
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    clear_weather = WeatherState(
        condition=WeatherCondition.CLEAR,
        rain_intensity=0.0,
        visibility_meters=250.0,
        friction_coefficient=WEATHER_CONFIG.clear_friction,
    )

    res = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=test_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.0,
        crowd_observation=neutral_crowd,
        weather_state=clear_weather,
    )
    assert res.target_speed_kmh == 40.0
    assert res.weather_safety_speed_kmh == 40.0


def test_light_rain_reduces_speed(test_segment, neutral_crowd):
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    light_rain = WeatherState(
        condition=WeatherCondition.LIGHT_RAIN,
        rain_intensity=0.35,
        visibility_meters=120.0,
        friction_coefficient=WEATHER_CONFIG.light_rain_friction,
    )

    res = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=test_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.0,
        crowd_observation=neutral_crowd,
        weather_state=light_rain,
    )
    # Expected: 40 * 0.80 = 32.0 km/h
    assert pytest.approx(res.weather_safety_speed_kmh, 0.1) == 32.0
    assert res.target_speed_kmh <= 35.0
    assert res.limiting_factor == "weather_condition"


def test_heavy_rain_reduces_speed_further(test_segment, neutral_crowd):
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    heavy_rain = WeatherState(
        condition=WeatherCondition.HEAVY_RAIN,
        rain_intensity=0.90,
        visibility_meters=45.0,
        friction_coefficient=WEATHER_CONFIG.heavy_rain_friction,
    )

    res = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=test_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.0,
        crowd_observation=neutral_crowd,
        weather_state=heavy_rain,
    )
    # Expected: 40 * 0.50 = 20.0 km/h, further restricted by visibility < 80m
    assert res.weather_safety_speed_kmh <= 20.0
    assert res.target_speed_kmh <= 25.0
    assert res.limiting_factor == "weather_condition"


def test_poor_visibility_increases_damping(test_segment, neutral_crowd):
    """Verify that severely degraded fog/downpour visibility modifies speed and headway limits."""
    controller = AdaptiveSpeedController(base_speed_kmh=40.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=40.0 / 3.6)

    dense_fog_weather = WeatherState(
        condition=WeatherCondition.CLEAR,
        rain_intensity=0.0,
        visibility_meters=35.0,  # Extreme low visibility
        friction_coefficient=0.85,
    )

    res = controller.compute_target_speed(
        vehicle=vehicle,
        current_segment=test_segment,
        pothole_safety_speed_kmh=40.0,
        historical_crowd_score=0.0,
        crowd_observation=neutral_crowd,
        weather_state=dense_fog_weather,
    )
    assert res.weather_safety_speed_kmh < 25.0
    assert any("visibility" in r.lower() for r in res.reasons)


def test_deterministic_mock_weather_transitions():
    """Verify scripted timeline transitions: CLEAR -> LIGHT_RAIN -> HEAVY_RAIN -> CLEAR."""
    provider = DeterministicMockWeatherProvider()

    # At t = 5.0s -> CLEAR
    w1 = provider.get_weather(sim_time_seconds=5.0)
    assert w1.condition == WeatherCondition.CLEAR
    assert w1.provider_mode == WeatherProviderMode.MOCK

    # At t = 20.0s -> LIGHT_RAIN
    w2 = provider.get_weather(sim_time_seconds=20.0)
    assert w2.condition == WeatherCondition.LIGHT_RAIN

    # At t = 40.0s -> HEAVY_RAIN
    w3 = provider.get_weather(sim_time_seconds=40.0)
    assert w3.condition == WeatherCondition.HEAVY_RAIN

    # At t = 60.0s -> Back to CLEAR
    w4 = provider.get_weather(sim_time_seconds=60.0)
    assert w4.condition == WeatherCondition.CLEAR


def test_missing_api_key_activates_mock_mode():
    """Verify that when no API key is provided, system safely and explicitly activates MOCK mode."""
    provider = OpenWeatherMapProvider(api_key=None)
    assert provider.mode == WeatherProviderMode.MOCK

    state = provider.get_weather()
    assert state.provider_mode == WeatherProviderMode.MOCK


def test_invalid_api_response_graceful_fallback():
    """Verify that HTTP 500 error, network timeout, or bad payload does not crash the system."""
    provider = OpenWeatherMapProvider(api_key="mock_invalid_key_999")

    # Mock requests.get returning HTTP 500 Server Error
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("requests.get", return_value=mock_resp):
        state = provider.get_weather()
        assert state is not None
        assert state.provider_mode == WeatherProviderMode.MOCK

    # Mock requests.get raising Timeout exception
    with patch("requests.get", side_effect=Exception("Connection timed out")):
        state_timeout = provider.get_weather()
        assert state_timeout is not None
        assert state_timeout.provider_mode == WeatherProviderMode.MOCK
