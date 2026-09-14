"""Simulation environment orchestrating the 12-step autonomous driving decision pipeline."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math
import time

from src.models import (
    VehicleState,
    RoadNetwork,
    RoadSegment,
    Waypoint,
    Pothole,
    WeatherState,
    CrowdObservation,
    SpeedArbitrationResult,
)
from src.perception.pothole_detector import PotholeDetector, SimulationPotholeDetector
from src.perception.crowd_provider import CrowdDensityProvider, ConfiguredCrowdProvider
from src.perception.weather_provider import WeatherProvider, DeterministicMockWeatherProvider
from src.memory.road_memory import RoadMemoryStore, SQLiteRoadMemoryStore
from src.planning.global_planner import GlobalRoutePlanner, RoutePlan
from src.planning.local_planner import LocalPathPlanner, LocalPlanResult
from src.control.speed_controller import AdaptiveSpeedController
from src.control.vehicle_controller import VehicleController, ControlCommand
from src.simulation.logger import SimulationLogger, StepLogRecord, TripSummary
from src.config import SPEED_CONFIG, ROAD_MEMORY_CONFIG


def create_default_indian_road_network() -> Tuple[RoadNetwork, List[Pothole]]:
    """
    Constructs a realistic representative road network modeling unstructured Indian road conditions.
    Topology:
      - Junction A: Start point (0, 0)
      - Junction B: Destination point (200, 0)
      - Junction D: Bypass waypoint (100, 40)
      
      Route Option 1 (Direct Main Road, 'seg_main_arterial'):
        - Distance: 200m, speed limit: 40 km/h.
        - Contains 3 severe potholes and moderate bazaar crowd.
      
      Route Option 2 (Ring Bypass Route):
        - Segment 'seg_bypass_leg1' (A -> D): 108m, speed limit: 50 km/h, clean surface, zero crowd.
        - Segment 'seg_bypass_leg2' (D -> B): 108m, speed limit: 50 km/h, clean surface, zero crowd.
        - Total distance: ~216m.
    """
    network = RoadNetwork()

    # 1. Main Arterial Segment (A -> B)
    main_waypoints: List[Waypoint] = []
    for s in range(0, 205, 5):
        main_waypoints.append(Waypoint(x=float(s), y=0.0, s=float(s)))

    seg_main = RoadSegment(
        id="seg_main_arterial",
        start_node="Node_A",
        end_node="Node_B",
        length_meters=200.0,
        speed_limit_kmh=40.0,
        lane_width_meters=4.2,
        historical_crowd_score=0.75,
        waypoints=main_waypoints,
    )
    network.add_segment(seg_main)

    # 2. Bypass Leg 1 (A -> D)
    d_x, d_y = 100.0, 40.0
    leg1_len = math.hypot(d_x, d_y)
    leg1_waypoints: List[Waypoint] = []
    steps = 40
    for i in range(steps + 1):
        ratio = i / steps
        s_dist = ratio * leg1_len
        leg1_waypoints.append(Waypoint(x=ratio * d_x, y=ratio * d_y, s=s_dist))

    seg_leg1 = RoadSegment(
        id="seg_bypass_leg1",
        start_node="Node_A",
        end_node="Node_D",
        length_meters=round(leg1_len, 1),
        speed_limit_kmh=50.0,
        lane_width_meters=4.5,
        historical_crowd_score=0.10,
        waypoints=leg1_waypoints,
    )
    network.add_segment(seg_leg1)

    # 3. Bypass Leg 2 (D -> B)
    dx2 = 200.0 - d_x
    dy2 = 0.0 - d_y
    leg2_len = math.hypot(dx2, dy2)
    leg2_waypoints: List[Waypoint] = []
    for i in range(steps + 1):
        ratio = i / steps
        s_dist = ratio * leg2_len
        leg2_waypoints.append(Waypoint(x=d_x + ratio * dx2, y=d_y + ratio * dy2, s=s_dist))

    seg_leg2 = RoadSegment(
        id="seg_bypass_leg2",
        start_node="Node_D",
        end_node="Node_B",
        length_meters=round(leg2_len, 1),
        speed_limit_kmh=50.0,
        lane_width_meters=4.5,
        historical_crowd_score=0.05,
        waypoints=leg2_waypoints,
    )
    network.add_segment(seg_leg2)

    # Ground truth potholes located on seg_main_arterial
    ground_truth_potholes = [
        Pothole(
            id="pothole_km_035",
            x=35.0,
            y=0.2,  # Near center of lane
            road_segment_id="seg_main_arterial",
            severity=0.85,
            confidence=0.92,
            radius=0.7,
        ),
        Pothole(
            id="pothole_km_085",
            x=85.0,
            y=-0.3,
            road_segment_id="seg_main_arterial",
            severity=0.75,
            confidence=0.88,
            radius=0.6,
        ),
        Pothole(
            id="pothole_km_140",
            x=140.0,
            y=0.1,
            road_segment_id="seg_main_arterial",
            severity=0.90,
            confidence=0.95,
            radius=0.8,
        ),
    ]

    return network, ground_truth_potholes


class SimulationEnvironment:
    """
    The orchestrator executing the 12-step autonomous decision cycle at fixed dt.
    Adheres strictly to the required operational sequence.
    """

    def __init__(
        self,
        network: RoadNetwork,
        ground_truth_potholes: List[Pothole],
        pothole_detector: Optional[PotholeDetector] = None,
        road_memory: Optional[RoadMemoryStore] = None,
        crowd_provider: Optional[CrowdDensityProvider] = None,
        weather_provider: Optional[WeatherProvider] = None,
        dt: float = 0.1,
        verbose_logging: bool = True,
    ):
        self.network = network
        self.ground_truth_potholes = ground_truth_potholes
        self.dt = dt

        # Interfaces & Subsystems
        self.pothole_detector = pothole_detector or SimulationPotholeDetector()
        self.road_memory = road_memory or SQLiteRoadMemoryStore()
        self.crowd_provider = crowd_provider or ConfiguredCrowdProvider()
        self.weather_provider = weather_provider or DeterministicMockWeatherProvider()

        self.global_planner = GlobalRoutePlanner(
            network=self.network,
            memory_store=self.road_memory,
        )
        self.local_planner = LocalPathPlanner()
        self.speed_controller = AdaptiveSpeedController()
        self.vehicle_controller = VehicleController()
        self.logger = SimulationLogger(verbose=verbose_logging)

    def run_trip(
        self,
        trip_id: str,
        start_node: str = "Node_A",
        end_node: str = "Node_B",
        consider_road_memory: bool = False,
        max_duration_s: float = 40.0,
    ) -> TripSummary:
        """
        Executes a complete trip from start_node to end_node.
        - trip_id: "trip_1" (exploration/discovery) or "trip_2" (memory-informed routing)
        - consider_road_memory: whether global planner incorporates past pothole risk
        """
        # Global Route Planning at trip departure
        route_plan: RoutePlan = self.global_planner.plan_route(
            start_node=start_node,
            end_node=end_node,
            consider_road_memory=consider_road_memory,
        )

        # Vehicle initialization at start of first segment
        first_segment = route_plan.segments[0]
        init_wp = first_segment.waypoints[0]
        next_wp = first_segment.waypoints[1]
        init_heading = math.atan2(next_wp.y - init_wp.y, next_wp.x - init_wp.x)

        vehicle = VehicleState(
            x=init_wp.x,
            y=init_wp.y,
            heading=init_heading,
            velocity=SPEED_CONFIG.min_safe_speed_kmh / 3.6,  # Start at rolling speed
            acceleration=0.0,
            steering_angle=0.0,
        )

        sim_time = 0.0
        step_index = 0
        current_segment_idx = 0
        newly_stored_potholes = 0
        deduplicated_potholes = 0
        collisions = 0
        detected_potholes_all: Dict[str, Pothole] = {}

        while sim_time < max_duration_s and current_segment_idx < len(route_plan.segments):
            curr_seg = route_plan.segments[current_segment_idx]
            
            # 1. Read simulation state: determine vehicle longitudinal coordinate s on segment
            curr_s = self._estimate_s_on_segment(vehicle, curr_seg)

            # Check if reached end of segment
            if curr_s >= curr_seg.length_meters - 2.0:
                current_segment_idx += 1
                if current_segment_idx >= len(route_plan.segments):
                    break
                curr_seg = route_plan.segments[current_segment_idx]
                curr_s = 0.0

            # 2. Read sensor/perception data
            # 3. Detect potholes
            detected_potholes = self.pothole_detector.detect(
                vehicle=vehicle,
                ground_truth_potholes=self.ground_truth_potholes,
            )

            # 4. Update road memory
            for p in detected_potholes:
                if p.id not in detected_potholes_all:
                    detected_potholes_all[p.id] = p
                is_new = self.road_memory.store_pothole(p, trip_id=trip_id)
                if is_new:
                    newly_stored_potholes += 1
                else:
                    deduplicated_potholes += 1

            # 5. Load historical road risks
            historical_pothole_risk = self.road_memory.calculate_segment_risk(curr_seg.id)

            # 6. Read crowd information
            historical_crowd = self.crowd_provider.get_historical_density(curr_seg.id)
            crowd_obs = self.crowd_provider.get_current_observation(
                vehicle=vehicle,
                current_segment=curr_seg,
                vehicle_s=curr_s,
            )

            # 7. Read weather information
            weather_state = self.weather_provider.get_weather(sim_time)

            # 8. Global route is active (route_plan already computed)

            # 9. Generate local collision-free path
            local_plan: LocalPlanResult = self.local_planner.plan(
                vehicle=vehicle,
                current_segment=curr_seg,
                detected_potholes=detected_potholes,
                current_s=curr_s,
            )

            # Check for physical collision with any pothole
            for p in self.ground_truth_potholes:
                if p.road_segment_id == curr_seg.id:
                    d = math.hypot(vehicle.x - p.x, vehicle.y - p.y)
                    if d < (p.radius + vehicle.width / 4.0):
                        collisions += 1

            # 10. Calculate adaptive target speed
            arbitration: SpeedArbitrationResult = self.speed_controller.compute_target_speed(
                vehicle=vehicle,
                current_segment=curr_seg,
                pothole_safety_speed_kmh=local_plan.pothole_safety_speed_kmh,
                historical_crowd_score=historical_crowd,
                crowd_observation=crowd_obs,
                weather_state=weather_state,
                dt=self.dt,
            )

            # 11. Send vehicle-control command
            cmd = self.vehicle_controller.compute_control(
                vehicle=vehicle,
                trajectory=local_plan.trajectory,
                target_speed_ms=arbitration.target_speed_ms,
                weather=weather_state,
            )
            # Step dynamics
            vehicle = self.vehicle_controller.step_dynamics(vehicle, cmd, dt=self.dt)

            # 12. Log decision and result
            log_record = StepLogRecord(
                step_index=step_index,
                sim_time_s=round(sim_time, 2),
                segment_id=curr_seg.id,
                vehicle_x=vehicle.x,
                vehicle_y=vehicle.y,
                vehicle_speed_kmh=vehicle.speed_kmh,
                target_speed_kmh=arbitration.target_speed_kmh,
                limiting_factor=arbitration.limiting_factor,
                weather_condition=weather_state.condition.value,
                weather_mode=weather_state.provider_mode.value,
                historical_crowd_score=historical_crowd,
                current_crowd_density=crowd_obs.current_density,
                detected_potholes_count=len(detected_potholes),
                active_avoidance=local_plan.active_avoidance,
                evading_pothole_id=local_plan.evading_pothole_id,
                chosen_lateral_offset_m=local_plan.chosen_lateral_offset_m,
                explanations=arbitration.reasons,
            )
            self.logger.log_step(log_record)

            sim_time += self.dt
            step_index += 1

        summary = self.logger.generate_trip_summary(
            trip_id=trip_id,
            route_segments=route_plan.segment_ids,
            route_reason=route_plan.selection_reason,
            is_alternate=route_plan.is_alternate_recommendation,
            potholes_encountered=len(self.ground_truth_potholes),
            potholes_detected=len(detected_potholes_all),
            potholes_new=newly_stored_potholes,
            potholes_dedup=deduplicated_potholes,
            collisions=collisions,
        )

        return summary

    def _estimate_s_on_segment(self, vehicle: VehicleState, segment: RoadSegment) -> float:
        """Finds closest waypoint arc-length s."""
        if not segment.waypoints:
            return 0.0
        min_dist = float("inf")
        closest_s = 0.0
        for wp in segment.waypoints:
            d = math.hypot(vehicle.x - wp.x, vehicle.y - wp.y)
            if d < min_dist:
                min_dist = d
                closest_s = wp.s
        return closest_s
