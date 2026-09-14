# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

A modular simulation prototype implementing three Smart Features (SPs) for autonomous navigation under challenging road conditions:
1. **SP1 — Pothole Perception, Local Avoidance, Road Memory & Route Re-planning**:
   - Simulation-based forward perception cone.
   - Sampling-based Frenet local swerving and collision avoidance.
   - Persistent SQLite road-memory hazard database with spatial deduplication ($\le 2$m radius).
   - Global route risk re-evaluation and alternate route recommendation on Trip 2.
2. **SP2 — Crowd-Aware Adaptive Speed**:
   - Integration of historical segment crowd density and real-time pedestrian observations.
   - Transparent physics/distance-based speed arbitration.
   - Gradual deceleration on crowd approach and automatic recovery upon exit.
3. **SP3 — Weather-Aware Adaptive Speed**:
   - Deterministic offline simulation mode (`CLEAR -> LIGHT_RAIN -> HEAVY_RAIN -> CLEAR`).
   - Live Weather API mode (`OpenWeatherMapProvider`) with graceful fallback to mock mode if unconfigured or on network failure.
   - Rain-intensity speed reduction, visibility damping, and friction-limited deceleration bounds.

---

## 1. Quickstart Commands

### Running Automated Test Suite
```powershell
python -m pytest -v tests/
```

### Running Full Integrated Demonstration
```powershell
python run_demo.py --all
```

### Running Specific Feature Demos

#### SP1: First Trip (Exploration & Pothole Memory Creation)
```powershell
python run_demo.py --trip 1
```

#### SP1: Second Trip (Road Memory Utilization & Alternate Route Recommendation)
```powershell
python run_demo.py --trip 2
```

#### SP2: Crowd-Aware Speed Adaptation & Recovery
```powershell
python run_demo.py --crowd
```

#### SP3: Deterministic Offline Weather Transitions
```powershell
python run_demo.py --weather-mock
```

#### SP3: Live Weather API Mode
```powershell
$env:OPENWEATHER_API_KEY="your_api_key_here"  # Optional
python run_demo.py --weather-live
```

---

## 2. Project Architecture & Directory Layout

```
.
├── data/
│   └── road_memory.db          # Persistent SQLite database for road memory hazards
├── docs/
│   ├── repository_audit.md     # Phase 0 repository audit
│   └── architecture.md         # Phase 1 architecture & interface specifications
├── src/
│   ├── config.py               # Centralized constants, thresholds & safety limits
│   ├── models.py               # Domain dataclasses & schemas
│   ├── control/
│   │   ├── speed_controller.py   # Rule-based multi-factor speed arbitration
│   │   └── vehicle_controller.py # Kinematic bicycle model & pure-pursuit steering
│   ├── memory/
│   │   └── road_memory.py        # SQLite persistent storage with spatial deduplication
│   ├── perception/
│   │   ├── crowd_provider.py     # Historical & dynamic crowd providers
│   │   ├── pothole_detector.py   # Forward perception cone detector
│   │   └── weather_provider.py   # Deterministic mock & live API weather providers
│   ├── planning/
│   │   ├── global_planner.py     # Graph route planner with hazard risk weighting
│   │   └── local_planner.py      # Frenet sampling trajectory collision avoidance
│   └── simulation/
│       ├── logger.py             # Structured audit logger & trip evaluator
│       └── simulation.py         # 12-step discrete-time simulation orchestrator
├── tests/
│   ├── test_integration.py     # Multi-hazard concurrent arbitration tests
│   ├── test_sp1_pothole.py      # Pothole detection, persistence & avoidance tests
│   ├── test_sp2_crowd.py        # Crowd density & recovery tests
│   └── test_sp3_weather.py      # Rain, visibility & API fallback tests
├── run_demo.py                 # CLI demonstration runner
└── README.md
```

---

## 3. Speed Arbitration Formula

The vehicle target speed is arbitrated using the strict formula:

$$\text{target\_speed} = \min(v_{\text{base}}, v_{\text{pothole\_safety}}, v_{\text{crowd\_safety}}, v_{\text{weather\_safety}}, v_{\text{road\_limit}})$$

Subject to the mandatory minimum safety limit:

$$\text{target\_speed} \ge v_{\text{min\_safe}} = 10.0\text{ km/h}$$

Every decision is accompanied by a structured explanation citing the contributing hazards.
