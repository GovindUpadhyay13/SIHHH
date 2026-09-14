"""Structured simulation logger and evaluation report generator."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output if console supports reconfiguration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger("SimulationEngine")


@dataclass
class StepLogRecord:
    """Log record for a single simulation time step."""
    step_index: int
    sim_time_s: float
    segment_id: str
    vehicle_x: float
    vehicle_y: float
    vehicle_speed_kmh: float
    target_speed_kmh: float
    limiting_factor: str
    weather_condition: str
    weather_mode: str
    historical_crowd_score: float
    current_crowd_density: float
    detected_potholes_count: int
    active_avoidance: bool
    evading_pothole_id: Optional[str]
    chosen_lateral_offset_m: float
    explanations: List[str]


@dataclass
class TripSummary:
    """Summary metrics of a completed simulation trip."""
    trip_id: str
    total_steps: int
    duration_s: float
    total_distance_m: float
    average_speed_kmh: float
    min_speed_kmh: float
    max_speed_kmh: float
    potholes_encountered: int
    potholes_detected: int
    potholes_stored_new: int
    potholes_deduplicated: int
    selected_route_segments: List[str]
    route_selection_reason: str
    is_alternate_route: bool
    near_misses_or_collisions: int


class SimulationLogger:
    """Collects step records, prints formatted live terminal summaries, and generates trip reports."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_records: List[StepLogRecord] = []

    def log_step(self, record: StepLogRecord) -> None:
        self.step_records.append(record)
        if self.verbose and (record.step_index % 10 == 0 or record.active_avoidance or record.limiting_factor != "base_cruise"):
            status_tag = f"[{record.weather_mode}|{record.weather_condition}]"
            speed_info = f"Speed: {record.vehicle_speed_kmh:4.1f} -> Target: {record.target_speed_kmh:4.1f} km/h ({record.limiting_factor})"
            avoidance_info = f"Avoid offset: {record.chosen_lateral_offset_m:+.2f}m" if record.active_avoidance else "Tracking center"
            print(f"t={record.sim_time_s:5.1f}s | {record.segment_id:17} | {status_tag:15} | {speed_info:48} | {avoidance_info}")
            if record.explanations and record.limiting_factor != "base_cruise":
                for reason in record.explanations[:2]:
                    # Safe ASCII prefix
                    print(f"       -> Reason: {reason}")

    def generate_trip_summary(
        self,
        trip_id: str,
        route_segments: List[str],
        route_reason: str,
        is_alternate: bool,
        potholes_encountered: int,
        potholes_detected: int,
        potholes_new: int,
        potholes_dedup: int,
        collisions: int,
    ) -> TripSummary:
        if not self.step_records:
            return TripSummary(
                trip_id=trip_id,
                total_steps=0,
                duration_s=0.0,
                total_distance_m=0.0,
                average_speed_kmh=0.0,
                min_speed_kmh=0.0,
                max_speed_kmh=0.0,
                potholes_encountered=potholes_encountered,
                potholes_detected=potholes_detected,
                potholes_stored_new=potholes_new,
                potholes_deduplicated=potholes_dedup,
                selected_route_segments=route_segments,
                route_selection_reason=route_reason,
                is_alternate_route=is_alternate,
                near_misses_or_collisions=collisions,
            )

        speeds = [r.vehicle_speed_kmh for r in self.step_records]
        duration = self.step_records[-1].sim_time_s
        total_dist = sum(r.vehicle_speed_kmh * (1000.0 / 3600.0) * 0.1 for r in self.step_records)

        return TripSummary(
            trip_id=trip_id,
            total_steps=len(self.step_records),
            duration_s=round(duration, 2),
            total_distance_m=round(total_dist, 1),
            average_speed_kmh=round(sum(speeds) / len(speeds), 1),
            min_speed_kmh=round(min(speeds), 1),
            max_speed_kmh=round(max(speeds), 1),
            potholes_encountered=potholes_encountered,
            potholes_detected=potholes_detected,
            potholes_stored_new=potholes_new,
            potholes_deduplicated=potholes_dedup,
            selected_route_segments=route_segments,
            route_selection_reason=route_reason,
            is_alternate_route=is_alternate,
            near_misses_or_collisions=collisions,
        )

    def export_json(self, output_path: Path | str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.step_records], f, indent=2)
