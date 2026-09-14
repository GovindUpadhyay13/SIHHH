# Phase 0: Repository Audit

## 1. Repository Structure
The workspace root (`c:\Users\govin\SIHHH`) was inspected.
- **Files present:** 0 files initially. The repository is completely clean / empty at start.
- **Version Control:** Git version `2.52.0.windows.1` detected.
- **Existing Frameworks / Simulators:** No simulator (such as CARLA, AirSim, LGSVL, RoadRunner, or Unreal Engine) is installed in the repository workspace.

## 2. Detected Language and Runtime Environment
- **Operating System:** Windows 11 (PowerShell environment)
- **Primary Runtime:** Python 3.12.7
- **Secondary Runtime:** Node.js v24.12.0
- **Package Manager:** pip 26.1.2

## 3. Detected Simulator and Sensors
- **Simulator Platform:** None pre-existing.
- **Sensors:** No physical hardware or ROS sensor topics configured.
- **Decision:** Build a self-contained, high-fidelity 2D Kinematic / Physics autonomous driving simulation engine in Python with deterministic discrete-time simulation (`dt = 0.1s`), modular sensor simulation (range-limited LiDAR / forward-facing camera perception cones), road networks with coordinate frames, and optional interactive visualizer.

## 4. Available Dependencies
Inspected Python 3.12 environment has:
- `pytest` (8.2.1) - Available for automated test suites.
- `numpy` (2.5.2) - Available for vector math, trajectory generation, spline interpolation, and distance transforms.
- `pydantic` (2.13.4) - Available for strict data schema validation.
- `requests` (2.32.5) - Available for live Weather API requests.
- `sqlite3` (Python Standard Library) - Available for persistent road memory database.
- `dataclasses`, `typing`, `logging`, `pathlib`, `json`, `math` (Python Standard Library).
- `pygame` (2.6.1) - Available for real-time visual simulation rendering if invoked.

## 5. Existing Executable Commands and Tests
- **Existing Commands:** None.
- **Existing Tests:** None.
- **Existing Entry Points:** None.

## 6. Missing Requirements and Gaps
1. No pre-trained pothole YOLO/PyTorch checkpoint exists in the repository.
   - *Rule applied:* Do NOT silently download or fabricate fake model weights.
   - *Decision:* Implement a clear `PotholeDetector` abstraction with a verified deterministic simulation perception detector (ground truth + noise/confidence modeling within field of view) and an explicit pluggable interface ready for vision models if provided.
2. No live weather API key is pre-configured in the environment.
   - *Decision:* Implement `WeatherProvider` protocol with both `DeterministicMockWeatherProvider` (default offline mode supporting clear -> light rain -> heavy rain -> clear transitions) and `OpenWeatherMapProvider` (enabled when `OPENWEATHER_API_KEY` is provided via environment, with graceful timeout, error handling, and mock fallback).
3. No pre-existing road network or crowd dataset.
   - *Decision:* Construct a structured road network representation supporting multi-segment graphs (nodes, edges, lane widths, speed limits) modeling unstructured Indian road conditions (variable road quality, crowd density zones, unexpected obstacles).

## 7. Implementation Risks and Mitigations
- **Risk:** Conflating global route planning with local collision avoidance.
  - *Mitigation:* Explicit separation: Global Route Planner operates on topological road graph (Dijkstra / A* with historical segment risk weights); Local Path Planner operates in continuous Frenet / Cartesian coordinates using trajectory sampling / polynomial collision avoidance.
- **Risk:** Arbitrary / non-explainable speed drops.
  - *Mitigation:* Implement transparent, rule-based arbitration with explicit component logging:
    `target_speed = min(base_speed, pothole_safety_speed, crowd_safety_speed, weather_safety_speed, road_speed_limit)`.
- **Risk:** Pothole database deduplication bugs causing duplicate records across trips.
  - *Mitigation:* Spatial deduplication threshold (Euclidean tolerance radius $\epsilon \approx 2.0$m) within road segment ID queries in SQLite.

## 8. Anti-Hallucination Disclosures
- **Evidence Found in Repository:** Clean directory, Python 3.12.7, numpy 2.5.2, pytest 8.2.1, sqlite3 built-in.
- **Assumption:** Standalone Python simulation is the designated platform since CARLA / ROS are neither installed nor configured.
- **Decision:** Implement cleanly structured, fully tested modular Python package `ad_simulator` with zero fake dependencies.
- **Limitation:** Visual/sensor perception uses geometric ray/frustum projection and synthetic detection uncertainty rather than heavy real-time neural network inference on missing video feeds.
