"""Transparent, rule-based adaptive speed controller with multi-factor safety arbitration."""

from __future__ import annotations
from typing import List, Tuple
import math

from src.models import (
    VehicleState,
    RoadSegment,
    CrowdObservation,
    WeatherState,
    WeatherCondition,
    SpeedArbitrationResult,
)
from src.config import SPEED_CONFIG, CROWD_CONFIG, WEATHER_CONFIG


class AdaptiveSpeedController:
    """
    Arbitrates target vehicle speed across multiple constraints:
    target_speed = min(base_speed, pothole_safety_speed, crowd_safety_speed, weather_safety_speed, road_speed_limit)
    subject to: target_speed >= min_safe_speed.

    Provides fully explainable audit records detailing every contributing hazard.
    """

    def __init__(
        self,
        base_speed_kmh: float = SPEED_CONFIG.base_cruise_speed_kmh,
        min_safe_speed_kmh: float = SPEED_CONFIG.min_safe_speed_kmh,
        max_speed_kmh: float = SPEED_CONFIG.max_vehicle_speed_kmh,
    ):
        self.base_speed_kmh = base_speed_kmh
        self.min_safe_speed_kmh = min_safe_speed_kmh
        self.max_speed_kmh = max_speed_kmh

    def compute_target_speed(
        self,
        vehicle: VehicleState,
        current_segment: RoadSegment,
        pothole_safety_speed_kmh: float,
        historical_crowd_score: float,
        crowd_observation: CrowdObservation,
        weather_state: WeatherState,
        dt: float = 0.1,
    ) -> SpeedArbitrationResult:
        """
        Computes the target speed by evaluating each safety constraint and finding the most restrictive condition:
        target_speed = minimum(base_speed, pothole_safety_speed, crowd_safety_speed, weather_safety_speed, road_speed_limit)
        """
        # 1. Road limit constraint
        road_limit_kmh = min(self.max_speed_kmh, current_segment.speed_limit_kmh)

        # 2. Base cruise speed
        base_kmh = min(self.base_speed_kmh, road_limit_kmh)

        # 3. Pothole safety speed (from local path planner)
        pothole_kmh = min(base_kmh, pothole_safety_speed_kmh)

        # 4. Crowd safety speed
        crowd_kmh, crowd_score, crowd_reasons = self._calculate_crowd_safe_speed(
            base_kmh=base_kmh,
            historical_score=historical_crowd_score,
            obs=crowd_observation,
        )

        # 5. Weather safety speed
        weather_kmh, weather_reasons = self._calculate_weather_safe_speed(
            base_kmh=base_kmh,
            weather=weather_state,
        )

        # 6. Minimum arbitration
        raw_candidates = {
            "road_speed_limit": road_limit_kmh,
            "pothole_avoidance": pothole_kmh,
            "crowd_density": crowd_kmh,
            "weather_condition": weather_kmh,
            "base_cruise": base_kmh,
        }

        limiting_factor = min(raw_candidates, key=raw_candidates.get)
        arbitrated_speed = raw_candidates[limiting_factor]

        # Enforce absolute safety floor
        final_speed_kmh = max(self.min_safe_speed_kmh, arbitrated_speed)
        final_speed_kmh = round(final_speed_kmh, 1)
        final_speed_ms = round(final_speed_kmh / 3.6, 2)

        # Compile comprehensive explanation
        reasons: List[str] = []
        if pothole_kmh < base_kmh:
            reasons.append(f"Pothole hazard detected ahead (safe speed: {pothole_kmh:.1f} km/h)")
        if crowd_reasons:
            reasons.extend(crowd_reasons)
        if weather_reasons:
            reasons.extend(weather_reasons)
        if road_limit_kmh < self.base_speed_kmh:
            reasons.append(f"Road segment speed limit enforced ({road_limit_kmh:.1f} km/h)")
        if not reasons:
            reasons.append("Optimal road, weather, and traffic conditions; cruising at standard base speed.")

        return SpeedArbitrationResult(
            base_speed_kmh=base_kmh,
            road_speed_limit_kmh=road_limit_kmh,
            pothole_safety_speed_kmh=round(pothole_kmh, 1),
            crowd_safety_speed_kmh=round(crowd_kmh, 1),
            weather_safety_speed_kmh=round(weather_kmh, 1),
            target_speed_kmh=final_speed_kmh,
            target_speed_ms=final_speed_ms,
            limiting_factor=limiting_factor,
            reasons=reasons,
        )

    def _calculate_crowd_safe_speed(
        self,
        base_kmh: float,
        historical_score: float,
        obs: CrowdObservation,
    ) -> Tuple[float, float, List[str]]:
        """
        Calculates safe speed given historical crowd index and current real-time crowd observation.
        Considers distance to crowded region for smooth deceleration and recovery.
        """
        reasons: List[str] = []
        # Weighted combination of historical and real-time observation
        combined_score = (
            CROWD_CONFIG.historical_weight * historical_score +
            CROWD_CONFIG.current_weight * obs.current_density
        )

        if combined_score <= 0.15 and not obs.is_active:
            return base_kmh, combined_score, reasons

        # Distance weighting: as distance shrinks from warning_dist to critical_dist, impact ramps to 1.0
        d = obs.distance_to_crowd_meters
        d_warn = CROWD_CONFIG.warning_distance_m
        d_crit = CROWD_CONFIG.critical_distance_m

        if d > d_warn:
            # Distant crowd: historical pre-braking only if segment is heavily crowded
            distance_factor = 0.20 if historical_score > 0.5 else 0.0
        elif d <= d_crit:
            distance_factor = 1.0
        else:
            distance_factor = 0.20 + 0.80 * ((d_warn - d) / (d_warn - d_crit))

        # Speed drop is proportional to combined score and proximity
        min_crowd_speed = CROWD_CONFIG.min_speed_in_dense_crowd_kmh
        speed_drop = (base_kmh - min_crowd_speed) * combined_score * distance_factor
        safe_crowd_speed = max(min_crowd_speed, base_kmh - speed_drop)

        reasons.append(
            f"Crowd awareness active: combined crowd score {combined_score:.2f} "
            f"(historical: {historical_score:.2f}, current: {obs.current_density:.2f}) "
            f"at distance {d:.1f}m -> crowd safe speed {safe_crowd_speed:.1f} km/h."
        )

        return safe_crowd_speed, combined_score, reasons

    def _calculate_weather_safe_speed(
        self,
        base_kmh: float,
        weather: WeatherState,
    ) -> Tuple[float, float, List[str]]:
        """
        Calculates weather-safe speed based on precipitation, surface friction, and visibility.
        """
        reasons: List[str] = []
        speed_factor = 1.0

        if weather.condition == WeatherCondition.LIGHT_RAIN:
            speed_factor = WEATHER_CONFIG.light_rain_speed_factor
            reasons.append(
                f"Light rain detected (intensity: {weather.rain_intensity:.2f}, "
                f"friction μ: {weather.friction_coefficient:.2f}) -> reduced speed to {speed_factor*100:.0f}%."
            )
        elif weather.condition == WeatherCondition.HEAVY_RAIN:
            speed_factor = WEATHER_CONFIG.heavy_rain_speed_factor
            reasons.append(
                f"Heavy rain detected (intensity: {weather.rain_intensity:.2f}, "
                f"friction μ: {weather.friction_coefficient:.2f}, visibility: {weather.visibility_meters:.0f}m) "
                f"-> reduced speed to {speed_factor*100:.0f}%."
            )

        # Visibility factor
        vis_factor = 1.0
        if weather.visibility_meters < 80.0:
            vis_factor = max(0.40, weather.visibility_meters / 100.0)
            reasons.append(f"Severely degraded visibility ({weather.visibility_meters:.0f}m) applies additional damping.")

        total_factor = min(speed_factor, vis_factor)
        safe_speed = max(self.min_safe_speed_kmh, base_kmh * total_factor)

        return safe_speed, reasons
