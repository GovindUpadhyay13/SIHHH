"""Global route planning across road network with road memory hazard cost integration."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import heapq

from src.models import RoadNetwork, RoadSegment
from src.memory.road_memory import RoadMemoryStore
from src.config import ROAD_MEMORY_CONFIG


@dataclass
class RoutePlan:
    """Represents an evaluated route from start to destination."""
    route_id: str
    segments: List[RoadSegment]
    total_length_m: float
    total_pothole_risk: float
    total_crowd_risk: float
    effective_cost: float
    is_alternate_recommendation: bool
    selection_reason: str

    @property
    def segment_ids(self) -> List[str]:
        return [s.id for s in self.segments]


class GlobalRoutePlanner:
    """
    Global topological route planner using graph search (Dijkstra).
    Considers road segment lengths, historical road-memory hazard risks, and crowd risks.
    """

    def __init__(
        self,
        network: RoadNetwork,
        memory_store: Optional[RoadMemoryStore] = None,
        pothole_weight: float = ROAD_MEMORY_CONFIG.pothole_risk_weight,
        risk_threshold: float = ROAD_MEMORY_CONFIG.route_replan_risk_threshold,
    ):
        self.network = network
        self.memory_store = memory_store
        self.pothole_weight = pothole_weight
        self.risk_threshold = risk_threshold

    def plan_route(
        self,
        start_node: str,
        end_node: str,
        consider_road_memory: bool = True,
    ) -> RoutePlan:
        """
        Plans the optimal route between start_node and end_node.
        If consider_road_memory is False (e.g., initial Trip 1), cost is pure geometric distance.
        If consider_road_memory is True (Trip 2), stored pothole risks increase segment traversal cost,
        triggering alternate route recommendation if risk exceeds the threshold.
        """
        # First compute baseline shortest geometric route
        baseline_path = self._find_path(start_node, end_node, use_hazard_costs=False)
        baseline_segments = [self.network.segments[seg_id] for seg_id in baseline_path]
        baseline_length = sum(s.length_meters for s in baseline_segments)
        baseline_pothole_risk = self._compute_pothole_risk(baseline_segments)
        baseline_crowd_risk = sum(s.historical_crowd_score for s in baseline_segments)

        if not consider_road_memory or self.memory_store is None:
            return RoutePlan(
                route_id="primary_route",
                segments=baseline_segments,
                total_length_m=baseline_length,
                total_pothole_risk=baseline_pothole_risk,
                total_crowd_risk=baseline_crowd_risk,
                effective_cost=baseline_length,
                is_alternate_recommendation=False,
                selection_reason="Trip 1 initial shortest path without historical road memory.",
            )

        # Trip 2 or memory-informed planning
        hazard_path = self._find_path(start_node, end_node, use_hazard_costs=True)
        hazard_segments = [self.network.segments[seg_id] for seg_id in hazard_path]
        hazard_length = sum(s.length_meters for s in hazard_segments)
        hazard_pothole_risk = self._compute_pothole_risk(hazard_segments)
        hazard_crowd_risk = sum(s.historical_crowd_score for s in hazard_segments)
        hazard_cost = self._compute_total_cost(hazard_segments, use_hazard_costs=True)

        # Check if baseline route exceeds risk threshold
        if baseline_pothole_risk >= self.risk_threshold and hazard_path != baseline_path:
            reason = (
                f"Primary route has high pothole risk ({baseline_pothole_risk:.2f} >= threshold {self.risk_threshold:.2f}). "
                f"Recommending alternate route via segments {hazard_path} with lower risk ({hazard_pothole_risk:.2f})."
            )
            return RoutePlan(
                route_id="alternate_route",
                segments=hazard_segments,
                total_length_m=hazard_length,
                total_pothole_risk=hazard_pothole_risk,
                total_crowd_risk=hazard_crowd_risk,
                effective_cost=hazard_cost,
                is_alternate_recommendation=True,
                selection_reason=reason,
            )

        return RoutePlan(
            route_id="primary_route",
            segments=hazard_segments,
            total_length_m=hazard_length,
            total_pothole_risk=hazard_pothole_risk,
            total_crowd_risk=hazard_crowd_risk,
            effective_cost=hazard_cost,
            is_alternate_recommendation=False,
            selection_reason=f"Primary route risk acceptable ({hazard_pothole_risk:.2f} < threshold {self.risk_threshold:.2f}).",
        )

    def _compute_pothole_risk(self, segments: List[RoadSegment]) -> float:
        if not self.memory_store:
            return 0.0
        return sum(self.memory_store.calculate_segment_risk(s.id) for s in segments)

    def _compute_total_cost(self, segments: List[RoadSegment], use_hazard_costs: bool) -> float:
        cost = 0.0
        for s in segments:
            base = s.length_meters
            if use_hazard_costs and self.memory_store:
                risk = self.memory_store.calculate_segment_risk(s.id)
                cost += base * (1.0 + self.pothole_weight * risk + 0.5 * s.historical_crowd_score)
            else:
                cost += base
        return cost

    def _find_path(self, start_node: str, end_node: str, use_hazard_costs: bool) -> List[str]:
        """Dijkstra graph search returning list of segment IDs."""
        distances: Dict[str, float] = {start_node: 0.0}
        previous_segment: Dict[str, Tuple[str, str]] = {}  # node -> (prev_node, segment_id)
        pq: List[Tuple[float, str]] = [(0.0, start_node)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_dist > distances.get(curr_node, float("inf")):
                continue
            if curr_node == end_node:
                break

            for seg_id in self.network.adjacency.get(curr_node, []):
                seg = self.network.segments[seg_id]
                neighbor = seg.end_node
                
                # Compute edge weight
                weight = seg.length_meters
                if use_hazard_costs and self.memory_store:
                    risk = self.memory_store.calculate_segment_risk(seg.id)
                    weight *= (1.0 + self.pothole_weight * risk + 0.5 * seg.historical_crowd_score)

                new_dist = curr_dist + weight
                if new_dist < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_dist
                    previous_segment[neighbor] = (curr_node, seg_id)
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        path: List[str] = []
        curr = end_node
        while curr in previous_segment:
            prev_node, seg_id = previous_segment[curr]
            path.append(seg_id)
            curr = prev_node
        path.reverse()
        return path
