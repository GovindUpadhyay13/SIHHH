"""Local path planning and collision avoidance around detected potholes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from src.models import VehicleState, Pothole, RoadSegment, TrajectoryPoint
from src.config import VEHICLE_CONFIG, SPEED_CONFIG


@dataclass
class LocalPlanResult:
    """Outcome of local collision-avoidance trajectory planning."""
    trajectory: List[TrajectoryPoint]
    pothole_safety_speed_kmh: float
    active_avoidance: bool
    evading_pothole_id: Optional[str]
    chosen_lateral_offset_m: float
    collision_warning: bool = False


class LocalPathPlanner:
    """
    Local path planner utilizing lateral offset trajectory sampling in Frenet coordinates.
    Generates smooth, collision-free local trajectories around detected potholes while respecting
    lane boundaries and kinematic constraints.
    """

    def __init__(
        self,
        vehicle_width_m: float = VEHICLE_CONFIG.width_m,
        planning_horizon_m: float = 30.0,
        lateral_samples: int = 15,
    ):
        self.vehicle_width_m = vehicle_width_m
        self.planning_horizon_m = planning_horizon_m
        self.lateral_samples = lateral_samples

    def plan(
        self,
        vehicle: VehicleState,
        current_segment: RoadSegment,
        detected_potholes: List[Pothole],
        current_s: float,
    ) -> LocalPlanResult:
        """
        Plans local trajectory ahead of vehicle.
        If potholes are detected ahead in the vehicle's lane path, evaluates lateral candidate
        offsets to avoid the hazard with safe clearance margin.
        """
        # Maximum lateral deviation allowed on this road segment (e.g. up to 1.5m from centerline)
        max_offset = min(1.6, max(0.8, (current_segment.lane_width_meters / 2.0) - 0.4))
        
        # Potholes ahead within planning horizon
        relevant_potholes: List[Pothole] = []
        for p in detected_potholes:
            dx = p.x - vehicle.x
            dy = p.y - vehicle.y
            forward_dist = dx * math.cos(vehicle.heading) + dy * math.sin(vehicle.heading)
            if 0.5 < forward_dist <= self.planning_horizon_m:
                relevant_potholes.append(p)

        # Generate candidate lateral offsets: from -max_offset to +max_offset
        step = (2 * max_offset) / max(1, self.lateral_samples - 1)
        candidate_offsets = [-max_offset + i * step for i in range(self.lateral_samples)]
        # Sort by absolute offset so centerline (offset=0) is preferred if free
        candidate_offsets.sort(key=lambda d: abs(d))

        best_trajectory: Optional[List[TrajectoryPoint]] = None
        best_offset = 0.0
        best_cost = float("inf")
        active_avoidance = False
        evading_id: Optional[str] = None
        closest_pothole_dist = float("inf")

        for d_offset in candidate_offsets:
            traj, min_clearance, hit_id, dist_to_hit = self._generate_and_evaluate_candidate(
                vehicle=vehicle,
                current_segment=current_segment,
                current_s=current_s,
                lateral_offset=d_offset,
                potholes=relevant_potholes,
            )

            # Minimum clearance required (lateral separation between trajectory point and pothole edge >= 0.15m)
            if min_clearance < 0.15:
                continue

            # Cost function: centerline deviation + penalty for proximity to obstacles
            cost = abs(d_offset) * 1.2 + (1.0 / max(0.2, min_clearance)) * 1.5

            if cost < best_cost:
                best_cost = cost
                best_trajectory = traj
                best_offset = d_offset
                if hit_id is not None:
                    active_avoidance = True
                    evading_id = hit_id
                    closest_pothole_dist = min(closest_pothole_dist, dist_to_hit)

        # Fallback if all offsets are blocked (extremely tight road / unavoidable pothole)
        if best_trajectory is None:
            best_trajectory, _, hit_id, dist_to_hit = self._generate_and_evaluate_candidate(
                vehicle=vehicle,
                current_segment=current_segment,
                current_s=current_s,
                lateral_offset=0.0,
                potholes=[],
            )
            best_offset = 0.0
            active_avoidance = True
            evading_id = relevant_potholes[0].id if relevant_potholes else None
            closest_pothole_dist = 5.0
            collision_warning = True
        else:
            collision_warning = False

        # Calculate pothole safety speed
        # If avoiding or approaching pothole within 15m, reduce speed proportionally
        if relevant_potholes:
            nearest_p = min(
                relevant_potholes,
                key=lambda p: math.hypot(p.x - vehicle.x, p.y - vehicle.y)
            )
            d_p = math.hypot(nearest_p.x - vehicle.x, nearest_p.y - vehicle.y)
            if d_p < 15.0:
                pothole_safety_speed = max(
                    SPEED_CONFIG.min_safe_speed_kmh,
                    25.0 - 10.0 * nearest_p.severity
                )
                active_avoidance = True
                evading_id = nearest_p.id
            else:
                pothole_safety_speed = SPEED_CONFIG.base_cruise_speed_kmh
        else:
            pothole_safety_speed = SPEED_CONFIG.base_cruise_speed_kmh

        if abs(best_offset) > 0.2:
            active_avoidance = True

        return LocalPlanResult(
            trajectory=best_trajectory,
            pothole_safety_speed_kmh=round(pothole_safety_speed, 1),
            active_avoidance=active_avoidance,
            evading_pothole_id=evading_id,
            chosen_lateral_offset_m=round(best_offset, 2),
            collision_warning=collision_warning,
        )

    def _generate_and_evaluate_candidate(
        self,
        vehicle: VehicleState,
        current_segment: RoadSegment,
        current_s: float,
        lateral_offset: float,
        potholes: List[Pothole],
    ) -> Tuple[List[TrajectoryPoint], float, Optional[str], float]:
        """Generates trajectory points and evaluates min clearance against potholes."""
        trajectory: List[TrajectoryPoint] = []
        num_points = 15
        ds = self.planning_horizon_m / num_points
        min_clearance = float("inf")
        closest_hit_id: Optional[str] = None
        closest_hit_dist = float("inf")

        for i in range(1, num_points + 1):
            s = current_s + i * ds
            center_x, center_y = current_segment.sample_point_at_s(s)
            
            # Road tangent heading
            next_x, next_y = current_segment.sample_point_at_s(s + 0.5)
            road_heading = math.atan2(next_y - center_y, next_x - center_x)
            
            # Lateral normal vector (-sin, cos)
            normal_x = -math.sin(road_heading)
            normal_y = math.cos(road_heading)

            # Rapid smooth transition to target lateral offset within 6m
            ramp = min(1.0, (i * ds) / 6.0)
            cur_d = lateral_offset * ramp

            pt_x = center_x + normal_x * cur_d
            pt_y = center_y + normal_y * cur_d

            # Clearance from center of vehicle trajectory to pothole center minus pothole radius
            # (Allows vehicle body / wheel track to bypass pothole with safe cushion)
            for p in potholes:
                dist = math.hypot(pt_x - p.x, pt_y - p.y)
                clearance = dist - (p.radius + 0.35)
                if clearance < min_clearance:
                    min_clearance = clearance
                    closest_hit_id = p.id
                    closest_hit_dist = math.hypot(vehicle.x - p.x, vehicle.y - p.y)

            trajectory.append(
                TrajectoryPoint(
                    x=pt_x,
                    y=pt_y,
                    target_speed_ms=vehicle.velocity,
                    heading=road_heading,
                    lateral_offset=cur_d,
                )
            )

        return trajectory, min_clearance, closest_hit_id, closest_hit_dist
