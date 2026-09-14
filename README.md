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

---

## 4. 3D Simulation (CARLA + Python) Integration Modules

Three modular, self-contained plug-ins connect into the existing decision and planning layer using a uniform integration interface:

### Module 1: Pothole Detection & Cost Map (`src/perception/pothole_cost_map.py`)
- **Showcase Scenario:** *Village Road / Unpaved Arterial* with irregular road surface damage.
- **Mechanism:** Ingests forward camera/LiDAR stream, applies YOLOv8 / perception model, and maintains a decaying 2D spatial cost overlay.
- **Cost Scaling:** Potholes raise traversal cost proportionally ($\text{cost} \propto \text{size} \times \text{confidence}$) rather than hard walls, naturally guiding smooth avoidance.
- **Thresholds & Rerouting:**
  - Decay half-life: `12.0s`
  - Influence radius: `2.5m`
  - Alternate route density threshold: `2.5` cost/point over a `25m` lookahead window, emitting a `suggest_alternate_route` callback.
- **Toggle Config:** `PotholeDetectorModule(enabled=True/False)`

### Module 2: Crowd-Aware Slowdown (`src/control/crowd_density_monitor.py`)
- **Showcase Scenario:** *Dense Market / Bazaar Area* with heavy pedestrian flow and informal crossings.
- **Mechanism:** Computes rolling density (agents per unit area over last $N=10$ frames) and maps to 4 discrete density bands:
  - `LOW` ($< 0.25$): Normal cruise speed ($40.0$ km/h)
  - `MEDIUM` ($0.25 - 0.55$): Speed cap $28.0$ km/h
  - `HIGH` ($0.55 - 0.80$): Speed cap $18.0$ km/h
  - `DENSE` ($> 0.80$): Speed cap $12.0$ km/h
- **Persistent Zone Cache:** Saves high-density coordinates to `data/crowd_zones_cache.json` for proactive slowing on approach across consecutive runs.
- **Toggle Config:** `CrowdDensityMonitor(enabled=True/False)`

### Module 3: Weather-Aware Slowdown (`src/control/weather_response.py`)
- **Showcase Scenario:** *Highway / Adverse Monsoon Conditions* with severe rain and reduced visibility.
- **Mechanism:** Direct telemetry integration with CARLA's `carla.WeatherParameters` (precipitation, fog density, wetness deposits, sun altitude).
- **Caution Levels & Speed Caps:**
  - `CLEAR`: $40.0$ km/h, Safety Margin $1.0\times$
  - `LIGHT_RAIN` ($\text{precip} \ge 15\%$ or $\text{wetness} \ge 30\%$): $32.0$ km/h, Safety Margin $1.3\times$
  - `HEAVY_RAIN_OR_FOG` ($\text{precip} \ge 50\%$ or $\text{fog} \ge 35\%$): $22.0$ km/h, Safety Margin $1.7\times$
  - `SEVERE` ($\text{precip} \ge 75\%$ or $\text{fog} \ge 70\%$): $14.0$ km/h, Safety Margin $2.2\times$
- **Toggle Config:** `WeatherResponseModule(enabled=True/False)`

---

## 5. Uniform Module Integration Protocol

All three modules output to the decision/planning layer with a uniform signature:
```python
# Planner Hook:
cost_map, speed_cap_pothole, reroute_event = pothole_module.update(vehicle, sensor_data, lookahead_path)

# Controller Hook (Speed Caps):
speed_cap_crowd, crowd_info = crowd_module.update(vehicle, perception_data)
speed_cap_weather, headway_mult, weather_info = weather_module.update(carla_weather)

# Transparent Minimum Arbitration:
active_caps = [c for c in [base_speed, speed_cap_pothole, speed_cap_crowd, speed_cap_weather] if c is not None]
commanded_speed = max(10.0, min(active_caps))
```

