"""Unit tests for SP1: Pothole Detection, Local Avoidance, Road Memory & Route Planning."""

import pytest
from pathlib import Path

from src.models import VehicleState, Pothole, RoadSegment, Waypoint
from src.perception.pothole_detector import SimulationPotholeDetector
from src.memory.road_memory import SQLiteRoadMemoryStore
from src.planning.global_planner import GlobalRoutePlanner
from src.planning.local_planner import LocalPathPlanner
from src.simulation.simulation import create_default_indian_road_network, SimulationEnvironment


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "road_memory_test.db"


def test_pothole_detection_output_format():
    """Verify pothole detection format, fields, and geometric visibility envelope."""
    detector = SimulationPotholeDetector(max_range_m=25.0, fov_deg=90.0)
    vehicle = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=5.0)

    # Pothole directly in front at 15m
    p1 = Pothole(id="p1", x=15.0, y=0.0, road_segment_id="s1", severity=0.8, confidence=0.9)
    # Pothole behind the vehicle (should NOT be detected)
    p2 = Pothole(id="p2", x=-10.0, y=0.0, road_segment_id="s1", severity=0.7, confidence=0.8)
    # Pothole beyond range at 50m
    p3 = Pothole(id="p3", x=50.0, y=0.0, road_segment_id="s1", severity=0.9, confidence=0.9)

    detections = detector.detect(vehicle, [p1, p2, p3])

    assert len(detections) == 1
    d = detections[0]
    assert d.id == "p1"
    assert isinstance(d.x, float)
    assert isinstance(d.y, float)
    assert 0.0 <= d.severity <= 1.0
    assert 0.0 <= d.confidence <= 1.0
    assert d.road_segment_id == "s1"
    assert d.radius > 0


def test_pothole_persistence_and_deduplication(temp_db_path):
    """Verify SQLite persistence and spatial deduplication within tolerance radius."""
    store = SQLiteRoadMemoryStore(db_path=temp_db_path, dedup_radius_m=2.0)
    store.clear()

    # First observation of a pothole at (35.0, 0.2)
    p_initial = Pothole(id="p_orig", x=35.0, y=0.2, road_segment_id="seg_main", severity=0.6, confidence=0.7)
    is_new_1 = store.store_pothole(p_initial, trip_id="trip_1")
    assert is_new_1 is True

    # Check persistence
    all_p = store.get_all_potholes()
    assert len(all_p) == 1
    assert all_p[0].id == "p_orig"

    # Second observation within 0.5m of the original pothole (should deduplicate)
    p_duplicate = Pothole(id="p_reobserved", x=35.3, y=0.4, road_segment_id="seg_main", severity=0.85, confidence=0.9)
    is_new_2 = store.store_pothole(p_duplicate, trip_id="trip_2")
    assert is_new_2 is False

    # Check that database still has only 1 pothole record with updated severity
    updated = store.get_all_potholes()
    assert len(updated) == 1
    assert updated[0].id == "p_orig"
    assert updated[0].severity == 0.85

    # Distinct pothole at 80.0m on same segment (beyond dedup radius)
    p_distinct = Pothole(id="p_far", x=80.0, y=-0.2, road_segment_id="seg_main", severity=0.7, confidence=0.8)
    is_new_3 = store.store_pothole(p_distinct, trip_id="trip_2")
    assert is_new_3 is True
    assert len(store.get_all_potholes()) == 2


def test_road_risk_calculation(temp_db_path):
    """Verify segment risk calculation: sum(severity * confidence)."""
    store = SQLiteRoadMemoryStore(db_path=temp_db_path)
    store.clear()

    assert store.calculate_segment_risk("seg_test") == 0.0

    store.store_pothole(
        Pothole(id="p1", x=10.0, y=0.0, road_segment_id="seg_test", severity=0.8, confidence=0.9),
        trip_id="trip_1",
    )
    store.store_pothole(
        Pothole(id="p2", x=30.0, y=0.0, road_segment_id="seg_test", severity=0.5, confidence=0.8),
        trip_id="trip_1",
    )

    # Expected: (0.8 * 0.9) + (0.5 * 0.8) = 0.72 + 0.40 = 1.12
    risk = store.calculate_segment_risk("seg_test")
    assert pytest.approx(risk, 0.01) == 1.12


def test_alternate_route_selection(temp_db_path):
    """Verify Trip 1 chooses direct route, Trip 2 switches to alternate bypass once risk exceeds threshold."""
    network, potholes = create_default_indian_road_network()
    store = SQLiteRoadMemoryStore(db_path=temp_db_path)
    store.clear()

    planner = GlobalRoutePlanner(network=network, memory_store=store, risk_threshold=1.0)

    # Trip 1: No road memory -> selects direct primary route
    plan_trip1 = planner.plan_route(start_node="Node_A", end_node="Node_B", consider_road_memory=False)
    assert plan_trip1.segment_ids == ["seg_main_arterial"]
    assert plan_trip1.is_alternate_recommendation is False

    # Store potholes from seg_main_arterial (risk > 1.0)
    for p in potholes:
        store.store_pothole(p, trip_id="trip_1")

    # Trip 2: Memory considered -> risk on seg_main_arterial exceeds 1.0 -> switches to bypass
    plan_trip2 = planner.plan_route(start_node="Node_A", end_node="Node_B", consider_road_memory=True)
    assert plan_trip2.segment_ids == ["seg_bypass_leg1", "seg_bypass_leg2"]
    assert plan_trip2.is_alternate_recommendation is True
    assert "recommending alternate route" in plan_trip2.selection_reason.lower()


def test_local_pothole_avoidance():
    """Verify local planner selects a collision-free lateral offset when pothole is in path."""
    planner = LocalPathPlanner()
    vehicle = VehicleState(x=20.0, y=0.0, heading=0.0, velocity=8.0)

    waypoints = [Waypoint(x=float(s), y=0.0, s=float(s)) for s in range(0, 100, 5)]
    segment = RoadSegment(
        id="s_test",
        start_node="A",
        end_node="B",
        length_meters=100.0,
        speed_limit_kmh=40.0,
        lane_width_meters=4.5,
        waypoints=waypoints,
    )

    # Pothole right in the centerline at x=30.0, y=0.0
    pothole = Pothole(id="p_mid", x=30.0, y=0.0, road_segment_id="s_test", severity=0.8, confidence=0.9, radius=0.35)

    result = planner.plan(
        vehicle=vehicle,
        current_segment=segment,
        detected_potholes=[pothole],
        current_s=20.0,
    )

    assert result.active_avoidance is True
    assert abs(result.chosen_lateral_offset_m) > 0.3  # Swerves away from center
    assert result.pothole_safety_speed_kmh < 40.0     # Reduces speed for safe maneuver
    assert result.collision_warning is False


def test_newly_discovered_pothole_handling(temp_db_path):
    """Verify newly discovered pothole in simulation is dynamically detected, avoided, and stored."""
    network, _ = create_default_indian_road_network()
    store = SQLiteRoadMemoryStore(db_path=temp_db_path)
    store.clear()

    # Create novel unseen pothole
    novel_pothole = Pothole(
        id="p_unseen_999",
        x=25.0,
        y=0.1,
        road_segment_id="seg_main_arterial",
        severity=0.9,
        confidence=0.95,
        radius=0.35,
    )

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=[novel_pothole],
        road_memory=store,
        verbose_logging=False,
    )

    summary = env.run_trip(trip_id="novel_test", consider_road_memory=False, max_duration_s=10.0)

    assert summary.potholes_detected >= 1
    assert summary.potholes_stored_new >= 1
    stored = store.get_all_potholes()
    assert any(p.id == "p_unseen_999" for p in stored)
