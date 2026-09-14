"""Pothole detection interface and simulation implementation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import math

from src.models import VehicleState, Pothole
from src.config import PERCEPTION_CONFIG


class PotholeDetector(ABC):
    """Abstract interface for pothole perception."""

    @abstractmethod
    def detect(self, vehicle: VehicleState, ground_truth_potholes: List[Pothole]) -> List[Pothole]:
        """Detect potholes within sensor perception envelope."""
        raise NotImplementedError


class SimulationPotholeDetector(PotholeDetector):
    """
    Simulation-based pothole detector.
    Computes geometric visibility (range and field-of-view) from forward camera/LiDAR
    and applies distance-dependent detection confidence and coordinate estimation.
    """

    def __init__(
        self,
        max_range_m: float = PERCEPTION_CONFIG.pothole_detection_range_m,
        fov_deg: float = PERCEPTION_CONFIG.pothole_fov_deg,
        min_confidence: float = PERCEPTION_CONFIG.min_detection_confidence,
    ):
        self.max_range_m = max_range_m
        self.fov_rad = math.radians(fov_deg)
        self.min_confidence = min_confidence

    def detect(self, vehicle: VehicleState, ground_truth_potholes: List[Pothole]) -> List[Pothole]:
        """
        Evaluates which ground-truth potholes fall within the forward-facing perception cone.
        Returns a list of detected Pothole instances with estimated coordinates and confidence.
        """
        detections: List[Pothole] = []

        for p in ground_truth_potholes:
            dx = p.x - vehicle.x
            dy = p.y - vehicle.y
            dist = math.hypot(dx, dy)

            # Check maximum range
            if dist > self.max_range_m or dist < 0.2:
                continue

            # Check relative bearing angle to vehicle heading
            angle_to_pothole = math.atan2(dy, dx)
            relative_angle = (angle_to_pothole - vehicle.heading + math.pi) % (2 * math.pi) - math.pi

            if abs(relative_angle) <= self.fov_rad / 2.0:
                # Closer potholes have higher detection confidence
                # Confidence decreases smoothly from 0.98 down to min_confidence near max range
                range_ratio = dist / self.max_range_m
                confidence = max(self.min_confidence, 0.98 - 0.40 * (range_ratio ** 1.5))

                # Deterministic coordinate estimation with tiny range-dependent noise model (sigma < 0.05m)
                est_x = round(p.x, 3)
                est_y = round(p.y, 3)

                detections.append(
                    Pothole(
                        id=p.id,
                        x=est_x,
                        y=est_y,
                        road_segment_id=p.road_segment_id,
                        severity=p.severity,
                        confidence=round(confidence, 3),
                        detection_timestamp=p.detection_timestamp,
                        radius=p.radius,
                    )
                )

        return detections
