"""Module 2: Crowd-Aware Slowdown & Persistent Known-Zone Monitor.

Reuses existing pedestrian detections from perception to estimate local crowd density
over rolling frame windows, classifies density into tunable bands (LOW, MEDIUM, HIGH, DENSE),
maintains persistent known crowd-heavy zones, and emits a smoothly interpolated target-speed-cap signal.
"""

from __future__ import annotations
import json
import time
import math
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.models import VehicleState
from src.config import DATA_DIR

logger = logging.getLogger("CrowdDensityMonitorModule")


class CrowdDensityBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    DENSE = "DENSE"


@dataclass
class CrowdZoneRecord:
    zone_id: str
    center_x: float
    center_y: float
    radius_m: float
    observed_frequency: int
    peak_density: float
    last_seen: float


@dataclass
class CrowdLogRecord:
    timestamp: float
    vehicle_pos: Tuple[float, float]
    agent_count: int
    density_score: float
    density_band: str
    target_speed_cap_kmh: float
    cache_hit: bool
    zone_id: Optional[str]


class CrowdDensityMonitor:
    """
    Module 2: Pluggable crowd density monitor.
    Computes rolling density, maps to bands, manages persistent crowd zone caching,
    and returns a smooth target-speed-cap signal.
    """

    def __init__(
        self,
        enabled: bool = True,
        forward_radius_m: float = 30.0,
        rolling_window_frames: int = 10,
        cache_file_path: Optional[Path | str] = None,
        # Tunable band thresholds (density score 0.0 to 1.0)
        medium_threshold: float = 0.25,
        high_threshold: float = 0.55,
        dense_threshold: float = 0.80,
        # Corresponding speed caps in km/h
        base_speed_kmh: float = 40.0,
        medium_speed_cap_kmh: float = 28.0,
        high_speed_cap_kmh: float = 18.0,
        dense_speed_cap_kmh: float = 12.0,
    ):
        self.enabled = enabled
        self.forward_radius_m = forward_radius_m
        self.rolling_window_frames = rolling_window_frames
        self.cache_file = Path(cache_file_path) if cache_file_path else DATA_DIR / "crowd_zones_cache.json"

        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.dense_threshold = dense_threshold

        self.base_speed_kmh = base_speed_kmh
        self.medium_speed_cap_kmh = medium_speed_cap_kmh
        self.high_speed_cap_kmh = high_speed_cap_kmh
        self.dense_speed_cap_kmh = dense_speed_cap_kmh

        self.rolling_counts: List[int] = []
        self._current_interpolated_cap: float = base_speed_kmh
        self.known_zones: Dict[str, CrowdZoneRecord] = {}
        self.logs: List[CrowdLogRecord] = []

        self._load_cache()

    def _load_cache(self) -> None:
        """Loads persistent known crowd-heavy zones from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.known_zones[k] = CrowdZoneRecord(**v)
                logger.info(f"Loaded {len(self.known_zones)} cached crowd-heavy zones.")
            except Exception as ex:
                logger.warning(f"Could not load crowd zones cache: {ex}")

    def _save_cache(self) -> None:
        """Saves known crowd-heavy zones to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.known_zones.items()}, f, indent=2)
        except Exception as ex:
            logger.warning(f"Could not persist crowd zones cache: {ex}")

    def update(
        self,
        vehicle: VehicleState,
        perception_data: Dict,  # Contains 'detected_agents' or 'crowd_observation'
        dt: float = 0.1,
    ) -> Tuple[Optional[float], Dict]:
        """
        Uniform integration tick:
        Returns:
          - speed_cap_kmh: Smoothly interpolated maximum allowable target speed.
          - telemetry_info: Dict containing band, count, cache hit, and zone data.
        """
        if not self.enabled:
            return None, {"enabled": False}

        now = time.time()

        # 1. Check existing pedestrian/agent observations
        raw_agents: List[Dict] = perception_data.get("detected_agents", [])
        crowd_obs = perception_data.get("crowd_observation")

        # Count agents within forward radius
        forward_count = 0
        if raw_agents:
            for a in raw_agents:
                ax = a.get("x", 0.0)
                ay = a.get("y", 0.0)
                dx = ax - vehicle.x
                dy = ay - vehicle.y
                dist = math.hypot(dx, dy)
                # Check forward facing
                heading_proj = dx * math.cos(vehicle.heading) + dy * math.sin(vehicle.heading)
                if 0.0 <= heading_proj <= self.forward_radius_m and dist <= self.forward_radius_m:
                    forward_count += 1
        elif crowd_obs:
            # Reusing crowd observation from current pipeline
            forward_count = int(crowd_obs.current_density * 25.0)

        # 2. Maintain rolling window frame count
        self.rolling_counts.append(forward_count)
        if len(self.rolling_counts) > self.rolling_window_frames:
            self.rolling_counts.pop(0)

        avg_count = sum(self.rolling_counts) / len(self.rolling_counts)
        # Normalize density: ~20 agents in forward radius corresponds to 1.0 density
        density_score = min(1.0, avg_count / 20.0)

        # 3. Check persistent known zones (Proactive slowing on approach)
        cache_hit = False
        matched_zone_id: Optional[str] = None
        for zid, zone in self.known_zones.items():
            dist_to_zone = math.hypot(zone.center_x - vehicle.x, zone.center_y - vehicle.y)
            if dist_to_zone <= zone.radius_m + 15.0:  # Within warning proximity
                cache_hit = True
                matched_zone_id = zid
                # If zone is known to be historically dense, boost effective density
                if zone.peak_density > density_score:
                    density_score = max(density_score, zone.peak_density * 0.85)
                break

        # 4. Map to Density Bands
        if density_score >= self.dense_threshold:
            band = CrowdDensityBand.DENSE
            target_cap = self.dense_speed_cap_kmh
        elif density_score >= self.high_threshold:
            band = CrowdDensityBand.HIGH
            target_cap = self.high_speed_cap_kmh
        elif density_score >= self.medium_threshold:
            band = CrowdDensityBand.MEDIUM
            target_cap = self.medium_speed_cap_kmh
        else:
            band = CrowdDensityBand.LOW
            target_cap = self.base_speed_kmh

        # 5. Cache update: if current density is high/dense, record zone
        if density_score >= self.high_threshold:
            zone_key = f"zone_{int(vehicle.x // 25) * 25}_{int(vehicle.y // 25) * 25}"
            if zone_key in self.known_zones:
                self.known_zones[zone_key].observed_frequency += 1
                self.known_zones[zone_key].peak_density = max(self.known_zones[zone_key].peak_density, density_score)
                self.known_zones[zone_key].last_seen = now
            else:
                self.known_zones[zone_key] = CrowdZoneRecord(
                    zone_id=zone_key,
                    center_x=round(vehicle.x, 1),
                    center_y=round(vehicle.y, 1),
                    radius_m=20.0,
                    observed_frequency=1,
                    peak_density=density_score,
                    last_seen=now,
                )
            self._save_cache()

        # 6. Smooth interpolation of target speed cap (avoids abrupt braking)
        interp_rate = 14.0 * dt  # km/h per second change rate
        if target_cap < self._current_interpolated_cap:
            # Smooth deceleration
            self._current_interpolated_cap = max(target_cap, self._current_interpolated_cap - interp_rate)
        else:
            # Smooth recovery acceleration
            self._current_interpolated_cap = min(target_cap, self._current_interpolated_cap + interp_rate * 0.75)

        speed_cap_result = round(self._current_interpolated_cap, 1)

        # Logging
        self.logs.append(
            CrowdLogRecord(
                timestamp=now,
                vehicle_pos=(round(vehicle.x, 1), round(vehicle.y, 1)),
                agent_count=forward_count,
                density_score=round(density_score, 2),
                density_band=band.value,
                target_speed_cap_kmh=speed_cap_result,
                cache_hit=cache_hit,
                zone_id=matched_zone_id,
            )
        )

        return speed_cap_result, {
            "density_score": round(density_score, 2),
            "density_band": band.value,
            "forward_agents": forward_count,
            "cache_hit": cache_hit,
            "zone_id": matched_zone_id,
            "speed_cap_kmh": speed_cap_result,
        }
