"""Crowd density providers for historical and real-time pedestrian density observation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import math

from src.models import CrowdObservation, VehicleState, RoadSegment
from src.config import CROWD_CONFIG


class CrowdDensityProvider(ABC):
    """Abstract interface for historical and real-time crowd observations."""

    @abstractmethod
    def get_historical_density(self, road_segment_id: str) -> float:
        """Returns configured historical crowd density index in [0.0, 1.0]."""
        raise NotImplementedError

    @abstractmethod
    def get_current_observation(
        self,
        vehicle: VehicleState,
        current_segment: RoadSegment,
        vehicle_s: float,
    ) -> CrowdObservation:
        """Returns real-time crowd observation ahead of vehicle."""
        raise NotImplementedError


class ConfiguredCrowdProvider(CrowdDensityProvider):
    """
    Configured/Simulated crowd provider.
    Maintains historical baseline crowd indices per road segment (e.g., bazaar, school zone)
    and models localized dynamic crowd clusters along road segments.
    """

    def __init__(
        self,
        historical_profiles: Optional[Dict[str, float]] = None,
        active_clusters: Optional[List[Dict[str, float]]] = None,
    ):
        # Configured historical crowd density index: 0.0 (deserted) to 1.0 (dense bazaar)
        self.historical_profiles: Dict[str, float] = historical_profiles or {
            "seg_market_main": 0.85,
            "seg_bypass_clean": 0.05,
            "seg_residential": 0.35,
            "seg_arterial_1": 0.20,
        }
        # Simulated dynamic pedestrian clusters: list of dicts with segment_id, s_center, density, radius
        self.active_clusters: List[Dict[str, float]] = active_clusters or []

    def add_dynamic_cluster(self, segment_id: str, s_center: float, density: float, radius: float = 15.0) -> None:
        """Add a localized dynamic crowd cluster along a road segment."""
        self.active_clusters.append({
            "segment_id": segment_id,
            "s_center": s_center,
            "density": min(1.0, max(0.0, density)),
            "radius": radius,
        })

    def clear_dynamic_clusters(self) -> None:
        self.active_clusters.clear()

    def get_historical_density(self, road_segment_id: str) -> float:
        """Returns configured historical crowd score (clearly labeled as configured simulation data)."""
        return self.historical_profiles.get(road_segment_id, 0.0)

    def get_current_observation(
        self,
        vehicle: VehicleState,
        current_segment: RoadSegment,
        vehicle_s: float,
    ) -> CrowdObservation:
        """
        Determines localized crowd density ahead of vehicle on current segment.
        Finds any dynamic cluster ahead within detection warning distance.
        """
        # Base observation from historical segment profile
        hist = self.get_historical_density(current_segment.id)
        
        # Check active clusters on this segment
        matching_clusters = [
            c for c in self.active_clusters
            if c["segment_id"] == current_segment.id and c["s_center"] >= (vehicle_s - 5.0)
        ]

        if not matching_clusters:
            # If no localized cluster, current observation reflects baseline segment density
            dist_to_hazard = max(0.0, current_segment.length_meters - vehicle_s)
            return CrowdObservation(
                road_segment_id=current_segment.id,
                current_density=round(hist * 0.5, 2),  # Ambient baseline
                distance_to_crowd_meters=round(dist_to_hazard, 1),
                is_active=hist > 0.3,
            )

        # Nearest cluster ahead
        nearest = min(matching_clusters, key=lambda c: c["s_center"] - vehicle_s)
        dist_ahead = max(0.0, nearest["s_center"] - vehicle_s)
        
        return CrowdObservation(
            road_segment_id=current_segment.id,
            current_density=nearest["density"],
            distance_to_crowd_meters=round(dist_ahead, 1),
            is_active=True,
        )
