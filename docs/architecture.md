# Phase 1: Autonomous Vehicle Simulation Architecture

## Project Title
**Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads**

---

## High-Level System Architecture

```mermaid
graph TD
    Sim[1. Simulation Environment] --> Sensors[2. Sensor / Perception Interface]
    Sensors --> PotholeDet[3. Pothole Detector]
    
    PotholeDet --> RoadMem[4. Road-Memory Database (SQLite)]
    RoadMem --> GlobalPlan[7. Global Route Planner]
    
    CrowdProv[5. Crowd-Density Provider] --> SpeedCtrl[9. Adaptive Speed Controller]
    WeatherProv[6. Weather Provider] --> SpeedCtrl
    
    GlobalPlan --> LocalPlan[8. Local Path Planner]
    PotholeDet --> LocalPlan
    Sensors --> LocalPlan
    
    SpeedCtrl --> VehCtrl[10. Vehicle Controller]
    LocalPlan --> VehCtrl
    
    VehCtrl --> Sim
    
    Sim --> LogEval[11. Logging & Evaluation Module]
    SpeedCtrl --> LogEval
    RoadMem --> LogEval
```

---

## 11 Core Components & Interfaces

### 1. Simulation Interface (`SimulationEnvironment`)
- **Role:** Coordinates discrete simulation time steps (`dt`), manages world state (road network, vehicle kinematic state, dynamic pedestrians, stationary potholes, environmental conditions).
- **State Vector:**
  - Vehicle: $[x, y, \theta, v, a, \delta]$ (Position, heading, speed, longitudinal acceleration, steering angle).
  - World time $t$, road segment index, active weather condition.

### 2. Sensor / Perception Interface (`SensorInterface`)
- **Role:** Models vehicle forward sensors (e.g. 120° FOV, 30m detection range).
- **Output:** `PerceptionFrame` containing ground-observed entities within line of sight, range, and field of view, with realistic measurement latency and noise model.

### 3. Pothole Detector (`PotholeDetector` Abstract Base Class)
- **Role:** Extracts pothole detections from sensor observations.
- **Implementations:**
  - `SimulationPotholeDetector`: Detects potholes when within perception envelope ($d \le R_{detect}$ and within angle $\phi \in [-\theta_{fov}/2, \theta_{fov}/2]$), calculating estimated 2D coordinates, severity $\in [0.1, 1.0]$, and confidence score $\in [0.0, 1.0]$.
  - `VisionPotholeDetector` (Interface placeholder for future YOLO/PyTorch models).

### 4. Road-Memory Database (`RoadMemoryStore`)
- **Role:** Persistent repository for historical road surface hazards.
- **Implementation:** `SQLiteRoadMemoryStore` utilizing SQLite local storage (`data/road_memory.db`).
- **Schema:**
  - `pothole_id` (TEXT PRIMARY KEY)
  - `road_segment_id` (TEXT)
  - `x` (REAL), `y` (REAL)
  - `severity` (REAL)
  - `confidence` (REAL)
  - `detection_timestamp` (TEXT)
  - `trip_id` (TEXT)
- **Deduplication:** Spatial clustering radius ($r \le 2.0$ meters) on the same road segment; updates confidence and severity rather than inserting duplicates.

### 5. Crowd-Density Provider (`CrowdDensityProvider`)
- **Role:** Provides both historical segment crowd levels and real-time localized crowd observations.
- **Components:**
  - Historical crowd profile ($D_{hist} \in [0.0, 1.0]$) per segment (e.g. market areas, transit stops).
  - Real-time crowd sensor observations ($D_{curr} \in [0.0, 1.0]$) and distance to crowd cluster $d_{crowd}$.
  - Configurable safety speed limits: $v_{min\_crowd} = 10 \text{ km/h}$, $v_{max} = 50 \text{ km/h}$.

### 6. Weather Provider (`WeatherProvider`)
- **Role:** Provides weather state and physical road-friction / visibility parameters.
- **Modes:**
  - `DeterministicMockWeatherProvider`: Deterministic sequence: `CLEAR -> LIGHT_RAIN -> HEAVY_RAIN -> CLEAR`.
  - `OpenWeatherMapProvider`: Network API client reading live precipitation, humidity, and visibility when an API key is present in environment variable `OPENWEATHER_API_KEY`. Graceful fallback to mock on timeout or network failure.
- **Output:** `WeatherState` (condition, rain_intensity $\in [0, 1]$, visibility_meters, surface_friction_coeff $\mu \in [0.3, 0.9]$).

### 7. Global Route Planner (`GlobalRoutePlanner`)
- **Role:** Selects the optimal topological route across the road network graph.
- **Cost Function:**
  $$C(e) = L_e \cdot \left(1.0 + w_{pothole} \cdot \text{Risk}_{pothole}(e) + w_{crowd} \cdot \text{Risk}_{crowd}(e)\right)$$
  - Trip 1: Initial trip with zero prior road memory; follows default fastest route.
  - Trip 2: Evaluates stored pothole risk from SQLite; if segment risk $> \text{threshold}$, automatically recommends alternate lower-risk detour.

### 8. Local Path Planner (`LocalPathPlanner`)
- **Role:** Collision avoidance and smooth trajectory generation around detected obstacles/potholes within the current road segment.
- **Algorithm:** Quintic polynomial / sampled lateral offset trajectory generation in Frenet coordinate frame ($s, d$), checking clearance against detected potholes and road boundaries.

### 9. Adaptive Speed Controller (`AdaptiveSpeedController`)
- **Role:** Transparent speed arbitration adhering to the strict safety constraint:
  $$v_{target} = \min(v_{base}, v_{pothole\_safe}, v_{crowd\_safe}, v_{weather\_safe}, v_{road\_limit})$$
  $$v_{target} \ge v_{min\_safe}$$
- Generates an explainable structured audit trail detailing exactly which constraint was active.

### 10. Vehicle Controller (`VehicleController`)
- **Role:** Kinematic bicycle model with longitudinal PID speed tracking and pure pursuit / Stanley steering control.
- **Commands:** Acceleration / deceleration $a \in [-4.0, 2.5] \text{ m/s}^2$, steering angle $\delta \in [-\delta_{max}, \delta_{max}]$.

### 11. Logging and Evaluation Module (`SimulationLogger`)
- **Role:** Logs step-by-step decisions, speeds, route evaluations, and outputs comparison metrics between Trip 1 and Trip 2.
