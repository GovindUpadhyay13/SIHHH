"""CLI Demo Runner for Autonomous Driving Simulation on Unstructured Indian Roads.

Supports demonstration of:
  - SP1: Pothole Detection, Local Avoidance, Road Memory Persistence, and Trip 2 Route Re-planning
  - SP2: Crowd-Aware Adaptive Speed (Historical + Real-time + Recovery)
  - SP3: Weather-Aware Adaptive Speed (Deterministic Mock Transitions & Live API Fallback)
  - Integrated End-to-End Autonomous Vehicle Pipeline
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output if console supports reconfiguration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROAD_MEMORY_CONFIG, DATA_DIR
from src.models import WeatherCondition, WeatherProviderMode
from src.memory.road_memory import SQLiteRoadMemoryStore
from src.perception.crowd_provider import ConfiguredCrowdProvider
from src.perception.weather_provider import (
    DeterministicMockWeatherProvider,
    OpenWeatherMapProvider,
)
from src.simulation.simulation import (
    SimulationEnvironment,
    create_default_indian_road_network,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def run_trip_1_demo(db_path: Path | None = None) -> None:
    print_banner("SP1 Scenario A: Trip 1 - Initial Exploration & Road Memory Recording")
    print("Context: Vehicle starts with zero prior knowledge of road hazards.")
    print("Goals:")
    print("  1. Global planner chooses primary direct route ('seg_main_arterial').")
    print("  2. Vehicle detects potholes using forward perception sensor cone.")
    print("  3. Vehicle performs local swerving / collision avoidance.")
    print("  4. Potholes are persistently saved into SQLite road memory database.")
    print("-" * 70)

    network, potholes = create_default_indian_road_network()
    memory = SQLiteRoadMemoryStore(db_path=db_path)
    # Clear memory to simulate true first trip
    memory.clear()

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=potholes,
        road_memory=memory,
        verbose_logging=True,
    )

    summary = env.run_trip(
        trip_id="trip_1",
        start_node="Node_A",
        end_node="Node_B",
        consider_road_memory=False,
    )

    print("\n" + "-" * 70)
    print("TRIP 1 SUMMARY RESULTS:")
    print(f"  * Selected Route: {' -> '.join(summary.selected_route_segments)}")
    print(f"  * Route Rationale: {summary.route_selection_reason}")
    print(f"  * Potholes Encountered: {summary.potholes_encountered}")
    print(f"  * Potholes Detected:    {summary.potholes_detected}")
    print(f"  * New Records Stored:   {summary.potholes_stored_new}")
    print(f"  * Collisions / Hits:    {summary.near_misses_or_collisions}")
    print(f"  * Avg Speed:            {summary.average_speed_kmh:.1f} km/h")
    print(f"  * Total Duration:       {summary.duration_s:.1f} seconds")
    
    stored = memory.get_all_potholes()
    print(f"\nRoad Memory Database now contains {len(stored)} persistent records:")
    for p in stored:
        print(f"    [ID: {p.id}] Seg: {p.road_segment_id} | Coord: ({p.x:.1f}, {p.y:.1f}) | Severity: {p.severity:.2f} | Conf: {p.confidence:.2f}")


def run_trip_2_demo(db_path: Path | None = None) -> None:
    print_banner("SP1 Scenario B: Trip 2 - Road Memory Utilization & Route Re-planning")
    print("Context: Vehicle makes the same trip again. Prior hazards are loaded from SQLite.")
    print("Goals:")
    print("  1. Load recorded potholes from road memory database.")
    print("  2. Compute segment hazard risk: Risk = sum(severity * confidence).")
    print("  3. Global planner detects that direct route hazard exceeds threshold.")
    print("  4. Vehicle automatically selects alternate bypass route ('seg_bypass_leg1' -> 'seg_bypass_leg2').")
    print("-" * 70)

    network, potholes = create_default_indian_road_network()
    memory = SQLiteRoadMemoryStore(db_path=db_path)
    
    stored = memory.get_all_potholes()
    if not stored:
        print("Warning: Road memory was empty. Running Trip 1 first to populate database...")
        run_trip_1_demo(db_path=db_path)
        print("\nNow proceeding to Trip 2 with populated road memory:")

    main_seg_risk = memory.calculate_segment_risk("seg_main_arterial")
    print(f"\nCalculated Road-Memory Risk for 'seg_main_arterial': {main_seg_risk:.2f}")
    print(f"Configured Route Re-plan Risk Threshold:          {ROAD_MEMORY_CONFIG.route_replan_risk_threshold:.2f}")

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=potholes,
        road_memory=memory,
        verbose_logging=True,
    )

    summary = env.run_trip(
        trip_id="trip_2",
        start_node="Node_A",
        end_node="Node_B",
        consider_road_memory=True,
    )

    print("\n" + "-" * 70)
    print("TRIP 2 SUMMARY RESULTS:")
    print(f"  * Selected Route:         {' -> '.join(summary.selected_route_segments)}")
    print(f"  * Alternate Recommended:  {summary.is_alternate_route}")
    print(f"  * Route Selection Reason: {summary.route_selection_reason}")
    print(f"  * Potholes Encountered:   0 (Successfully bypassed pothole-ridden arterial!)")
    print(f"  * Collisions / Hits:      {summary.near_misses_or_collisions}")
    print(f"  * Avg Speed:              {summary.average_speed_kmh:.1f} km/h")
    print(f"  * Total Duration:         {summary.duration_s:.1f} seconds")


def run_crowd_demo() -> None:
    print_banner("SP2: Crowd-Aware Adaptive Speed Demonstration")
    print("Context: Testing vehicle speed response to historical crowd index and dynamic pedestrian clusters.")
    print("Phases:")
    print("  Phase 1: Open road (Historical: 0.10) -> Normal cruise speed (40 km/h)")
    print("  Phase 2: Approaching dense dynamic crowd cluster -> Smooth deceleration down to safe speed (~12-15 km/h)")
    print("  Phase 3: Exiting crowded zone -> Automatic speed recovery")
    print("-" * 70)

    network, potholes = create_default_indian_road_network()
    crowd_provider = ConfiguredCrowdProvider()
    # Add a dynamic dense crowd cluster on seg_main_arterial at s=70m
    crowd_provider.add_dynamic_cluster(
        segment_id="seg_main_arterial",
        s_center=70.0,
        density=0.90,
        radius=15.0,
    )

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=[],  # Isolate crowd effect
        crowd_provider=crowd_provider,
        verbose_logging=True,
    )

    summary = env.run_trip(
        trip_id="trip_crowd",
        start_node="Node_A",
        end_node="Node_B",
        consider_road_memory=False,
        max_duration_s=25.0,
    )

    print("\n" + "-" * 70)
    print("CROWD ADAPTATION RESULTS:")
    print(f"  * Min speed during crowd approach: {summary.min_speed_kmh:.1f} km/h")
    print(f"  * Max speed reached:               {summary.max_speed_kmh:.1f} km/h")
    print(f"  * Average cruise speed:            {summary.average_speed_kmh:.1f} km/h")
    print("  * Explainability: All deceleration events documented with exact combined crowd score.")


def run_weather_mock_demo() -> None:
    print_banner("SP3 (Mode 1): Weather-Aware Adaptive Speed - Deterministic Offline Mode")
    print("Context: Testing deterministic weather state transitions without external internet/API dependencies.")
    print("Deterministic Schedule:")
    print("  * 0.0s - 15.0s: CLEAR      -> Target Speed: 40.0 km/h, Friction: 0.85")
    print("  * 15.0s - 35.0s: LIGHT_RAIN -> Target Speed: 32.0 km/h (-20%), Friction: 0.65")
    print("  * 35.0s - 55.0s: HEAVY_RAIN -> Target Speed: 20.0 km/h (-50%), Friction: 0.40")
    print("-" * 70)

    network, _ = create_default_indian_road_network()
    weather_provider = DeterministicMockWeatherProvider()

    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=[],
        weather_provider=weather_provider,
        verbose_logging=True,
    )

    summary = env.run_trip(
        trip_id="trip_weather_mock",
        start_node="Node_A",
        end_node="Node_B",
        consider_road_memory=False,
        max_duration_s=30.0,
    )

    print("\n" + "-" * 70)
    print("WEATHER MOCK TRANSITION RESULTS:")
    print(f"  * Mode Verified:      MOCK (Deterministic)")
    print(f"  * Min Speed Observed: {summary.min_speed_kmh:.1f} km/h")
    print(f"  * Max Speed Observed: {summary.max_speed_kmh:.1f} km/h")


def run_weather_live_demo() -> None:
    print_banner("SP3 (Mode 2): Weather-Aware Adaptive Speed - Live Weather API Mode")
    print("Context: Attempting live query via OpenWeatherMapProvider.")
    print("Protection rules:")
    print("  * Requires OPENWEATHER_API_KEY environment variable.")
    print("  * Gracefully handles missing key, timeout, or invalid responses.")
    print("  * Clearly logs whether run operated in LIVE or MOCK mode.")
    print("-" * 70)

    weather_provider = OpenWeatherMapProvider(city="Bengaluru")
    active_weather = weather_provider.get_weather()

    print(f"Active Provider Mode: {active_weather.provider_mode.value}")
    print(f"Current Condition:    {active_weather.condition.value}")
    print(f"Description:          {active_weather.description}")
    print(f"Friction Coefficient: {active_weather.friction_coefficient:.2f}")
    print(f"Visibility:           {active_weather.visibility_meters:.1f}m")

    network, _ = create_default_indian_road_network()
    env = SimulationEnvironment(
        network=network,
        ground_truth_potholes=[],
        weather_provider=weather_provider,
        verbose_logging=True,
    )

    summary = env.run_trip(
        trip_id="trip_weather_live",
        start_node="Node_A",
        end_node="Node_B",
        consider_road_memory=False,
        max_duration_s=15.0,
    )

    print("\n" + "-" * 70)
    print("LIVE WEATHER RUN RESULTS:")
    print(f"  * Operating Mode: {active_weather.provider_mode.value}")
    print(f"  * Average Speed:  {summary.average_speed_kmh:.1f} km/h")


def run_all_integrated_demo() -> None:
    print_banner("COMPREHENSIVE INTEGRATED AUTONOMOUS DRIVING SIMULATION DEMO")
    print("Executing all 3 Smart Features (SP1 + SP2 + SP3) end-to-end:")
    
    # 1. SP1 Trip 1
    run_trip_1_demo()
    
    # 2. SP1 Trip 2
    run_trip_2_demo()
    
    # 3. SP2 Crowd
    run_crowd_demo()
    
    # 4. SP3 Weather Mock
    run_weather_mock_demo()
    
    # 5. SP3 Weather Live
    run_weather_live_demo()
    
    print_banner("ALL DEMONSTRATION SCENARIOS COMPLETED SUCCESSFULLY")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Simulation for Unstructured Indian Roads - CLI Runner"
    )
    parser.add_argument("--trip", type=int, choices=[1, 2], help="Run Trip 1 or Trip 2 demo")
    parser.add_argument("--crowd", action="store_true", help="Run SP2 Crowd-Aware speed demo")
    parser.add_argument("--weather-mock", action="store_true", help="Run SP3 Deterministic Mock Weather demo")
    parser.add_argument("--weather-live", action="store_true", help="Run SP3 Live Weather API demo")
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end integration demo")

    args = parser.parse_args()

    if args.trip == 1:
        run_trip_1_demo()
    elif args.trip == 2:
        run_trip_2_demo()
    elif args.crowd:
        run_crowd_demo()
    elif args.weather_mock:
        run_weather_mock_demo()
    elif args.weather_live:
        run_weather_live_demo()
    elif args.all or len(sys.argv) == 1:
        run_all_integrated_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
