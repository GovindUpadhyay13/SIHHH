"""Kinematic bicycle model vehicle controller with longitudinal acceleration and pure-pursuit steering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from src.models import VehicleState, TrajectoryPoint, WeatherState
from src.config import VEHICLE_CONFIG


@dataclass
class ControlCommand:
    """Actuator commands sent to the vehicle kinematic model."""
    acceleration_mps2: float
    steering_rad: float
    target_speed_ms: float


class VehicleController:
    """
    Low-level controller implementing:
    - Pure Pursuit lateral path tracking.
    - Closed-loop longitudinal speed tracking with friction-aware acceleration bounds.
    - Discrete kinematic bicycle state update.
    """

    def __init__(
        self,
        wheelbase_m: float = VEHICLE_CONFIG.wheelbase_m,
        max_steer_rad: float = VEHICLE_CONFIG.max_steer_rad,
        max_accel_mps2: float = VEHICLE_CONFIG.max_accel_mps2,
        max_decel_mps2: float = VEHICLE_CONFIG.max_decel_mps2,
        lookahead_distance_m: float = 6.0,
    ):
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad
        self.max_accel_mps2 = max_accel_mps2
        self.max_decel_mps2 = max_decel_mps2
        self.lookahead_distance_m = lookahead_distance_m

    def compute_control(
        self,
        vehicle: VehicleState,
        trajectory: List[TrajectoryPoint],
        target_speed_ms: float,
        weather: WeatherState,
    ) -> ControlCommand:
        """Computes steering and acceleration commands."""
        # 1. Longitudinal control with friction-aware braking limit
        effective_max_decel = self.max_decel_mps2 * weather.friction_coefficient
        effective_max_accel = self.max_accel_mps2 * weather.friction_coefficient

        speed_error = target_speed_ms - vehicle.velocity
        kp_speed = 1.2
        desired_accel = kp_speed * speed_error
        accel = max(-effective_max_decel, min(effective_max_accel, desired_accel))

        # 2. Lateral control: Pure Pursuit
        steering = 0.0
        if trajectory:
            # Find target point at lookahead distance
            target_pt = self._find_lookahead_point(vehicle, trajectory)
            
            # Vector to lookahead point in vehicle local frame
            dx = target_pt.x - vehicle.x
            dy = target_pt.y - vehicle.y
            
            local_x = math.cos(-vehicle.heading) * dx - math.sin(-vehicle.heading) * dy
            local_y = math.sin(-vehicle.heading) * dx + math.cos(-vehicle.heading) * dy

            dist_sq = local_x ** 2 + local_y ** 2
            if dist_sq > 0.1:
                # Curvature kappa = 2 * local_y / Ld^2
                curvature = (2.0 * local_y) / dist_sq
                steering = math.atan(self.wheelbase_m * curvature)
                steering = max(-self.max_steer_rad, min(self.max_steer_rad, steering))

        return ControlCommand(
            acceleration_mps2=round(accel, 2),
            steering_rad=round(steering, 3),
            target_speed_ms=target_speed_ms,
        )

    def step_dynamics(
        self,
        vehicle: VehicleState,
        cmd: ControlCommand,
        dt: float = 0.1,
    ) -> VehicleState:
        """Integrates vehicle state using kinematic bicycle model."""
        # Kinematic bicycle equations:
        new_v = max(0.0, vehicle.velocity + cmd.acceleration_mps2 * dt)
        new_heading = vehicle.heading + (new_v / self.wheelbase_m) * math.tan(cmd.steering_rad) * dt
        # Normalize heading to [-pi, pi]
        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi

        new_x = vehicle.x + new_v * math.cos(vehicle.heading) * dt
        new_y = vehicle.y + new_v * math.sin(vehicle.heading) * dt

        return VehicleState(
            x=round(new_x, 3),
            y=round(new_y, 3),
            heading=round(new_heading, 4),
            velocity=round(new_v, 3),
            acceleration=cmd.acceleration_mps2,
            steering_angle=cmd.steering_rad,
            width=vehicle.width,
            length=vehicle.length,
        )

    def _find_lookahead_point(
        self,
        vehicle: VehicleState,
        trajectory: List[TrajectoryPoint],
    ) -> TrajectoryPoint:
        for pt in trajectory:
            dist = math.hypot(pt.x - vehicle.x, pt.y - vehicle.y)
            if dist >= self.lookahead_distance_m:
                return pt
        return trajectory[-1]
