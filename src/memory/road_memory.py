"""Persistent road memory database for road hazard tracking and deduplication."""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple
import sqlite3
import math
import time

from src.models import Pothole
from src.config import ROAD_MEMORY_CONFIG


class RoadMemoryStore(ABC):
    """Abstract interface for road-hazard persistence and risk queries."""

    @abstractmethod
    def store_pothole(self, pothole: Pothole, trip_id: str) -> bool:
        """Stores or updates a pothole. Returns True if newly inserted, False if deduplicated/updated."""
        raise NotImplementedError

    @abstractmethod
    def get_potholes_by_segment(self, road_segment_id: str) -> List[Pothole]:
        """Retrieve all recorded potholes for a specific road segment."""
        raise NotImplementedError

    @abstractmethod
    def get_all_potholes(self) -> List[Pothole]:
        """Retrieve all stored potholes across all road segments."""
        raise NotImplementedError

    @abstractmethod
    def calculate_segment_risk(self, road_segment_id: str) -> float:
        """Calculates aggregate pothole risk for a segment based on severity and confidence."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored road memory records."""
        raise NotImplementedError


class SQLiteRoadMemoryStore(RoadMemoryStore):
    """
    SQLite-backed implementation of road memory.
    Supports persistent storage, spatial deduplication within road segments,
    and hazard risk aggregation.
    Ensures safe explicit connection lifecycle management across all platforms.
    """

    def __init__(self, db_path: Optional[Path | str] = None, dedup_radius_m: float = ROAD_MEMORY_CONFIG.spatial_dedup_radius_m):
        self.db_path = Path(db_path) if db_path else ROAD_MEMORY_CONFIG.db_path
        self.dedup_radius_m = dedup_radius_m
        self._init_db()

    def _execute(self, query: str, params: Tuple = (), commit: bool = False) -> List[sqlite3.Row]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
            return cursor.fetchall()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes database schema if not already present."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS potholes (
                pothole_id TEXT PRIMARY KEY,
                road_segment_id TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                severity REAL NOT NULL,
                confidence REAL NOT NULL,
                detection_timestamp REAL NOT NULL,
                trip_id TEXT NOT NULL,
                radius REAL NOT NULL DEFAULT 0.35
            );
            """,
            commit=True,
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_potholes_segment
            ON potholes (road_segment_id);
            """,
            commit=True,
        )

    def store_pothole(self, pothole: Pothole, trip_id: str) -> bool:
        """
        Stores a pothole into persistent database.
        Applies spatial deduplication: if a known pothole exists on the same segment
        within dedup_radius_m, updates its severity, confidence, and timestamp.
        Returns True if a new record was created, False if deduplicated and updated.
        """
        existing = self.get_potholes_by_segment(pothole.road_segment_id)
        
        # Check for spatial proximity to an existing pothole
        closest_pothole: Optional[Pothole] = None
        min_dist = float("inf")

        for ep in existing:
            dist = math.hypot(ep.x - pothole.x, ep.y - pothole.y)
            if dist < min_dist and dist <= self.dedup_radius_m:
                min_dist = dist
                closest_pothole = ep

        if closest_pothole is not None:
            # Deduplicate and update
            updated_severity = max(closest_pothole.severity, pothole.severity)
            updated_confidence = min(1.0, max(closest_pothole.confidence, pothole.confidence) + 0.05)
            self._execute(
                """
                UPDATE potholes
                SET severity = ?,
                    confidence = ?,
                    detection_timestamp = ?,
                    trip_id = ?
                WHERE pothole_id = ?;
                """,
                (
                    updated_severity,
                    updated_confidence,
                    pothole.detection_timestamp,
                    trip_id,
                    closest_pothole.id,
                ),
                commit=True,
            )
            return False
        else:
            # Insert new pothole
            self._execute(
                """
                INSERT INTO potholes (
                    pothole_id, road_segment_id, x, y, severity, confidence, detection_timestamp, trip_id, radius
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    pothole.id,
                    pothole.road_segment_id,
                    pothole.x,
                    pothole.y,
                    pothole.severity,
                    pothole.confidence,
                    pothole.detection_timestamp,
                    trip_id,
                    pothole.radius,
                ),
                commit=True,
            )
            return True

    def get_potholes_by_segment(self, road_segment_id: str) -> List[Pothole]:
        """Retrieves all potholes recorded on a given road segment."""
        rows = self._execute(
            """
            SELECT pothole_id, road_segment_id, x, y, severity, confidence, detection_timestamp, radius
            FROM potholes
            WHERE road_segment_id = ?;
            """,
            (road_segment_id,),
        )
        return [
            Pothole(
                id=row["pothole_id"],
                road_segment_id=row["road_segment_id"],
                x=row["x"],
                y=row["y"],
                severity=row["severity"],
                confidence=row["confidence"],
                detection_timestamp=row["detection_timestamp"],
                radius=row["radius"],
            )
            for row in rows
        ]

    def get_all_potholes(self) -> List[Pothole]:
        """Retrieves all stored potholes across all segments."""
        rows = self._execute(
            """
            SELECT pothole_id, road_segment_id, x, y, severity, confidence, detection_timestamp, radius
            FROM potholes;
            """
        )
        return [
            Pothole(
                id=row["pothole_id"],
                road_segment_id=row["road_segment_id"],
                x=row["x"],
                y=row["y"],
                severity=row["severity"],
                confidence=row["confidence"],
                detection_timestamp=row["detection_timestamp"],
                radius=row["radius"],
            )
            for row in rows
        ]

    def calculate_segment_risk(self, road_segment_id: str) -> float:
        """
        Calculates aggregate hazard risk for a segment:
        Risk = sum(severity_i * confidence_i) for all potholes on the segment.
        """
        potholes = self.get_potholes_by_segment(road_segment_id)
        if not potholes:
            return 0.0
        return sum(p.severity * p.confidence for p in potholes)

    def clear(self) -> None:
        """Wipes the database table (useful for fresh tests or Trip 1 initialization)."""
        self._execute("DELETE FROM potholes;", commit=True)
