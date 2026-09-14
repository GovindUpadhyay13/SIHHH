"""Domain models and data structures for autonomous driving simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time


class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    LIGHT_RAIN = "LIGHT_RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"


class WeatherProviderMode(str, Enum):
    MOCK = "MOCK"
    LIVE = "LIVE"


@dataclass
class WeatherState:
    """Represents environmental weather state affecting vehicle dynamics and visibility."""
    condition: WeatherCondition
    rain_intensity: float  # 0.0 (dry) to 1.0 (torrential downpour)
    visibility_meters: float  # e.g., 200m down to 25m
    friction_coefficient: float  # e.g., 0.85 (dry asphalt) to 0.40 (wet/slick)
    provider_mode: WeatherProviderMode = WeatherProviderMode.MOCK
    temperature_celsius: Optional[float] = 28.0
    description: str = "Simulated clear weather"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Pothole:
    """Represents a physical road pothole or detected surface anomaly."""
    id: str
    x: float
    y: float
    road_segment_id: str
    severity: float  # 0.0 (minor) to 1.0 (severe wheel-damage risk)
    confidence: float  # 0.0 to 1.0
    detection_timestamp: float = field(default_factory=time.time)
    radius: float = 0.6  # Physical obstacle radius in meters


@dataclass
class CrowdObservation:
    """Represents localized crowd observation around or ahead of the vehicle."""
    road_segment_id: str
    current_density: float  # 0.0 (deserted) to 1.0 (dense pedestrian market)
    distance_to_crowd_meters: float  # Distance ahead of vehicle to crowd center
    is_active: bool = True


@dataclass
class VehicleState:
    """Represents the instantaneous kinematic/dynamic state of the vehicle."""
    x: float
    y: float
    heading: float  # radians (0 = positive X axis, counter-clockwise)
    velocity: float  # longitudinal speed in m/s (convert to km/h by multiplying by 3.6)
    acceleration: float = 0.0  # m/s^2
    steering_angle: float = 0.0  # radians
    width: float = 1.8  # vehicle width in meters
    length: float = 4.2  # vehicle length in meters

    @property
    def speed_kmh(self) -> float:
        return self.velocity * 3.6


@dataclass
class Waypoint:
    """A geometric point along a road segment centerline."""
    x: float
    y: float
    s: float  # Arc length distance from segment origin in meters


@dataclass
class RoadSegment:
    """Represents a directed link connecting two road junctions."""
    id: str
    start_node: str
    end_node: str
    length_meters: float
    speed_limit_kmh: float
    lane_width_meters: float = 4.0  # Single lane or shared lane typical of Indian roads
    historical_crowd_score: float = 0.0  # 0.0 (none) to 1.0 (perennial bazaar)
    waypoints: List[Waypoint] = field(default_factory=list)

    def sample_point_at_s(self, s: float) -> Tuple[float, float]:
        """Interpolate (x, y) along segment centerline at longitudinal distance s."""
        if not self.waypoints:
            return (0.0, 0.0)
        if s <= self.waypoints[0].s:
            return (self.waypoints[0].x, self.waypoints[0].y)
        if s >= self.waypoints[-1].s:
            return (self.waypoints[-1].x, self.waypoints[-1].y)
        
        for i in range(len(self.waypoints) - 1):
            w1 = self.waypoints[i]
            w2 = self.waypoints[i + 1]
            if w1.s <= s <= w2.s:
                ratio = (s - w1.s) / max(1e-5, (w2.s - w1.s))
                interp_x = w1.x + ratio * (w2.x - w1.x)
                interp_y = w1.y + ratio * (w2.y - w1.y)
                return (interp_x, interp_y)
        return (self.waypoints[-1].x, self.waypoints[-1].y)


@dataclass
class RoadNetwork:
    """Topological graph of road segments and junction nodes."""
    segments: Dict[str, RoadSegment] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # node -> outgoing segment ids

    def add_segment(self, segment: RoadSegment) -> None:
        self.segments[segment.id] = segment
        if segment.start_node not in self.adjacency:
            self.adjacency[segment.start_node] = []
        self.adjacency[segment.start_node].append(segment.id)


@dataclass
class SpeedArbitrationResult:
    """Transparent, explainable result of speed arbitration."""
    base_speed_kmh: float
    road_speed_limit_kmh: float
    pothole_safety_speed_kmh: float
    crowd_safety_speed_kmh: float
    weather_safety_speed_kmh: float
    target_speed_kmh: float
    target_speed_ms: float
    limiting_factor: str
    reasons: List[str]


@dataclass
class TrajectoryPoint:
    """Local trajectory waypoint with target kinematics."""
    x: float
    y: float
    target_speed_ms: float
    heading: float
    curvature: float = 0.0
    lateral_offset: float = 0.0
