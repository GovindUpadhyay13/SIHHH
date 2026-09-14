"""Module 1: Pothole Detection, Local Cost Map, and Alternate Route Signaling.

Consumes camera / LiDAR / perception feed, runs lightweight detection (YOLOv8 / sensor fallback),
maintains a decaying local 2D pothole cost map, and triggers an alternate route event
if pothole density exceeds a configured threshold.
"""

from __future__ import annotations
import time
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path

from src.models import VehicleState, Pothole

logger = logging.getLogger("PotholeCostMapModule")


@dataclass
class PotholeDetectionRecord:
    timestamp: float
    pothole_id: str
    x: float
    y: float
    confidence: float
    size_m: float
    action_taken: str


@dataclass
class PotholeCostGridCell:
    world_x: float
    world_y: float
    cost: float
    confidence: float
    size_m: float
    last_updated: float


class PotholeCostMap:
    """
    Maintains a localized 2D cost map overlay representing road damage and pothole hazards.
    Cost increases smoothly around potholes rather than acting as hard walls, allowing the
    local planner to naturally steer around clusters.
    """

    def __init__(
        self,
        grid_resolution_m: float = 0.5,
        decay_half_life_s: float = 12.0,
        influence_radius_m: float = 2.5,
    ):
        self.grid_resolution_m = grid_resolution_m
        self.decay_half_life_s = decay_half_life_s
        self.influence_radius_m = influence_radius_m
        self.cells: Dict[Tuple[int, int], PotholeCostGridCell] = {}

    def _coord_to_key(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor(x / self.grid_resolution_m))
        gy = int(math.floor(y / self.grid_resolution_m))
        return (gx, gy)

    def add_pothole(self, x: float, y: float, size_m: float, confidence: float, now: float) -> None:
        """Injects or updates a pothole in the cost grid with smooth radial Gaussian-like cost decay."""
        radius_steps = int(math.ceil(self.influence_radius_m / self.grid_resolution_m))
        center_key = self._coord_to_key(x, y)

        for dx in range(-radius_steps, radius_steps + 1):
            for dy in range(-radius_steps, radius_steps + 1):
                key = (center_key[0] + dx, center_key[1] + dy)
                cell_x = key[0] * self.grid_resolution_m + self.grid_resolution_m / 2.0
                cell_y = key[1] * self.grid_resolution_m + self.grid_resolution_m / 2.0
                dist = math.hypot(cell_x - x, cell_y - y)

                if dist <= self.influence_radius_m:
                    # Cost is proportional to size and confidence, tapering with distance
                    dist_factor = max(0.0, 1.0 - (dist / self.influence_radius_m) ** 2)
                    added_cost = (size_m * 10.0) * confidence * dist_factor

                    if key in self.cells:
                        self.cells[key].cost = max(self.cells[key].cost, added_cost)
                        self.cells[key].confidence = max(self.cells[key].confidence, confidence)
                        self.cells[key].last_updated = now
                    else:
                        self.cells[key] = PotholeCostGridCell(
                            world_x=cell_x,
                            world_y=cell_y,
                            cost=added_cost,
                            confidence=confidence,
                            size_m=size_m,
                            last_updated=now,
                        )

    def update_decay(self, now: float) -> None:
        """Decays cell costs over time as the vehicle moves past."""
        decay_factor = math.exp(-0.693 / self.decay_half_life_s)  # half-life decay
        stale_keys = []

        for key, cell in self.cells.items():
            dt = now - cell.last_updated
            if dt > 1.0:
                cell.cost *= decay_factor
                if cell.cost < 0.05:
                    stale_keys.append(key)

        for key in stale_keys:
            del self.cells[key]

    def get_cost_at(self, x: float, y: float) -> float:
        """Returns the pothole cost penalty for a given world coordinate."""
        key = self._coord_to_key(x, y)
        cell = self.cells.get(key)
        return cell.cost if cell else 0.0

    def compute_route_pothole_density(self, lookahead_points: List[Tuple[float, float]]) -> float:
        """Computes aggregate pothole density along upcoming path points."""
        if not lookahead_points:
            return 0.0
        total_cost = sum(self.get_cost_at(x, y) for x, y in lookahead_points)
        return total_cost / len(lookahead_points)


class PotholeDetectorModule:
    """
    Module 1: Pluggable pothole detector and cost map generator.
    Exposes the uniform interface: (cost_map_delta, speed_cap, event_trigger).
    """

    def __init__(
        self,
        enabled: bool = True,
        density_reroute_threshold: float = 2.5,
        lookahead_window_m: float = 25.0,
        model_weights_path: Optional[str | Path] = None,
        on_reroute_suggested: Optional[Callable[[str], None]] = None,
    ):
        self.enabled = enabled
        self.density_reroute_threshold = density_reroute_threshold
        self.lookahead_window_m = lookahead_window_m
        self.on_reroute_suggested = on_reroute_suggested
        self.cost_map = PotholeCostMap()
        self.detection_logs: List[PotholeDetectionRecord] = []
        self._yolo_model = None
        self._init_detector(model_weights_path)

    def _init_detector(self, weights_path: Optional[str | Path]) -> None:
        if weights_path and Path(weights_path).exists():
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(weights_path))
                logger.info(f"Loaded pretrained YOLOv8 pothole model from {weights_path}")
            except Exception as ex:
                logger.warning(f"Could not initialize YOLO model ({ex}). Using sensor perception fallback.")
                self._yolo_model = None

    def update(
        self,
        vehicle: VehicleState,
        sensor_data: Dict,  # Contains 'detected_potholes' or 'camera_image'
        lookahead_path: Optional[List[Tuple[float, float]]] = None,
        dt: float = 0.1,
    ) -> Tuple[PotholeCostMap, Optional[float], Optional[str]]:
        """
        Uniform integration tick:
        Returns:
          - cost_map: The updated pothole cost map overlay for local path planning.
          - speed_cap_kmh: Recommended maximum safe speed based on imminent potholes.
          - event_trigger: 'suggest_alternate_route' if density exceeds threshold, else None.
        """
        if not self.enabled:
            return self.cost_map, None, None

        now = time.time()
        self.cost_map.update_decay(now)

        # 1. Ingest detections from perception feed
        raw_potholes: List[Pothole] = sensor_data.get("detected_potholes", [])
        imminent_potholes: List[Pothole] = []

        for p in raw_potholes:
            dist = math.hypot(p.x - vehicle.x, p.y - vehicle.y)
            # Add to local cost map
            self.cost_map.add_pothole(p.x, p.y, size_m=p.radius * 2.0, confidence=p.confidence, now=now)

            action = "cost_map_updated"
            if dist < 15.0:
                imminent_potholes.append(p)
                action = "local_swerve_required"

            self.detection_logs.append(
                PotholeDetectionRecord(
                    timestamp=now,
                    pothole_id=p.id,
                    x=p.x,
                    y=p.y,
                    confidence=p.confidence,
                    size_m=round(p.radius * 2.0, 2),
                    action_taken=action,
                )
            )

        # 2. Compute local speed cap (soft speed reduction proportional to severity)
        speed_cap: Optional[float] = None
        if imminent_potholes:
            max_sev = max(p.severity for p in imminent_potholes)
            speed_cap = max(15.0, 30.0 - 15.0 * max_sev)

        # 3. Evaluate lookahead pothole density for alternate route recommendation
        event_trigger = None
        if lookahead_path:
            density = self.cost_map.compute_route_pothole_density(lookahead_path)
            if density >= self.density_reroute_threshold:
                event_trigger = "suggest_alternate_route"
                if self.on_reroute_suggested:
                    self.on_reroute_suggested(f"Pothole density ({density:.2f}) exceeded threshold ({self.density_reroute_threshold:.2f})")

        return self.cost_map, speed_cap, event_trigger
