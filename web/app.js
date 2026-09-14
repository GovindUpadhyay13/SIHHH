/**
 * Autonomous Driving Simulation — Interactive Web Visualizer
 * Implements SP1 (Potholes), SP2 (Crowd), SP3 (Weather) with real-time 2D Canvas rendering.
 */

// Road Network Definition (World Coordinates in Meters -> Canvas Pixels)
const WORLD_CONFIG = {
  scale: 3.8, // Pixels per meter
  originX: 80,
  originY: 340,
  mainLengthM: 200,
  bypassApexX: 100,
  bypassApexY: 55, // 55m north of main road
};

function worldToCanvas(x, y) {
  return {
    cx: WORLD_CONFIG.originX + x * WORLD_CONFIG.scale,
    cy: WORLD_CONFIG.originY - y * WORLD_CONFIG.scale,
  };
}

// Simulation State
const simState = {
  isRunning: true,
  currentTrip: 1,
  activeRoute: "seg_main_arterial", // or "seg_bypass"
  weather: "CLEAR", // CLEAR, LIGHT_RAIN, HEAVY_RAIN
  weatherFriction: 0.85,
  hasCrowdSurge: false,
  
  // Vehicle Kinematics
  veh: {
    s: 0.0, // Distance along current segment
    x: 0.0,
    y: 0.0,
    heading: 0.0,
    speedKmh: 10.0,
    targetSpeedKmh: 40.0,
    steerAngleRad: 0.0,
    lateralOffsetM: 0.0,
    activeAvoidance: false,
    evadingId: null,
  },

  // Ground Truth Potholes
  potholes: [
    { id: "pothole_km_035", x: 35.0, y: 0.2, severity: 0.85, detected: false, radius: 0.7 },
    { id: "pothole_km_085", x: 85.0, y: -0.3, severity: 0.75, detected: false, radius: 0.6 },
    { id: "pothole_km_140", x: 140.0, y: 0.1, severity: 0.90, detected: false, radius: 0.8 },
  ],

  // Road Memory Database (Simulated SQLite store)
  roadMemory: [],

  // Dynamic Crowd
  crowdCluster: {
    s: 70.0,
    density: 0.85,
    active: false,
    radiusM: 15.0,
  },

  // Multi-Hazard Arbitration State
  arbitration: {
    base: 40.0,
    roadLimit: 40.0,
    potholeSafe: 40.0,
    crowdSafe: 40.0,
    weatherSafe: 40.0,
    activeLimiter: "base_cruise",
    reasons: [],
  },

  pedestrians: [],
};

// Generate ambient crowd particles around bazaar
for (let i = 0; i < 28; i++) {
  simState.pedestrians.push({
    x: 60 + Math.random() * 25,
    y: -4 + (Math.random() - 0.5) * 8,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    size: 2 + Math.random() * 2,
  });
}

// Canvas & DOM Elements
const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
const rainOverlay = document.getElementById("rainOverlay");

const txtSpeed = document.getElementById("txt-speed");
const txtTargetSpeed = document.getElementById("txt-target-speed");
const speedBar = document.getElementById("speedBar");
const txtActiveLimiter = document.getElementById("txt-active-limiter");

const valBase = document.getElementById("val-base");
const valLimit = document.getElementById("val-limit");
const valPothole = document.getElementById("val-pothole");
const valCrowd = document.getElementById("val-crowd");
const valWeather = document.getElementById("val-weather");

const rowBase = document.getElementById("row-base");
const rowLimit = document.getElementById("row-limit");
const rowPothole = document.getElementById("row-pothole");
const rowCrowd = document.getElementById("row-crowd");
const rowWeather = document.getElementById("row-weather");

const memoryList = document.getElementById("memoryList");
const auditLog = document.getElementById("auditLog");
const chipMemory = document.getElementById("chip-memory");
const chipWeather = document.getElementById("chip-weather");

// Logging Helper
function logEvent(type, message) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  entry.textContent = `[${new Date().toISOString().substring(11, 19)}] ${message}`;
  auditLog.appendChild(entry);
  auditLog.scrollTop = auditLog.scrollHeight;
}

// Simulation Control Logic
function resetVehicle() {
  simState.veh.s = 0.0;
  simState.veh.speedKmh = 10.0;
  simState.veh.lateralOffsetM = 0.0;
  simState.veh.activeAvoidance = false;
  simState.veh.evadingId = null;

  if (simState.activeRoute === "seg_main_arterial") {
    simState.veh.x = 0.0;
    simState.veh.y = 0.0;
    simState.veh.heading = 0.0;
  } else {
    simState.veh.x = 0.0;
    simState.veh.y = 0.0;
    simState.veh.heading = Math.atan2(WORLD_CONFIG.bypassApexY, WORLD_CONFIG.bypassApexX);
  }
}

// Scenario: Run Trip 1
document.getElementById("btn-trip1").addEventListener("click", () => {
  simState.currentTrip = 1;
  simState.activeRoute = "seg_main_arterial";
  simState.potholes.forEach(p => p.detected = false);
  resetVehicle();
  simState.isRunning = true;
  logEvent("route", "TRIP 1: Started exploration on direct main arterial (Node A -> Node B). Road memory inactive.");
});

// Scenario: Run Trip 2
document.getElementById("btn-trip2").addEventListener("click", () => {
  simState.currentTrip = 2;
  const recordedRisk = simState.roadMemory.reduce((sum, p) => sum + p.severity * p.confidence, 0);
  
  if (recordedRisk >= 1.2 || simState.roadMemory.length >= 2) {
    simState.activeRoute = "seg_bypass";
    logEvent("route", `TRIP 2: Evaluated road memory! Arterial hazard risk (${recordedRisk.toFixed(2)} >= 1.20). Recommending alternate Bypass Route (Node A -> Node D -> Node B).`);
  } else {
    simState.activeRoute = "seg_main_arterial";
    logEvent("route", "TRIP 2: Insufficient road memory hazards recorded. Direct route acceptable.");
  }
  resetVehicle();
  simState.isRunning = true;
});

// Scenario: Dynamic Crowd Surge
document.getElementById("btn-crowd-surge").addEventListener("click", () => {
  simState.crowdCluster.active = !simState.crowdCluster.active;
  if (simState.crowdCluster.active) {
    logEvent("crowd", "SP2 ALERT: High pedestrian crowd surge active at market crossing (s=70m, density: 0.90).");
  } else {
    logEvent("crowd", "SP2: Pedestrian surge cleared. Resuming normal traffic density.");
  }
});

// Scenario: Weather Selector
document.getElementById("weatherSelect").addEventListener("change", (e) => {
  simState.weather = e.target.value;
  chipWeather.textContent = `WEATHER: ${simState.weather}`;
  rainOverlay.className = "rain-overlay";

  if (simState.weather === "CLEAR") {
    simState.weatherFriction = 0.85;
    logEvent("weather", "SP3: Weather transition -> CLEAR. Dry asphalt (Friction μ: 0.85).");
  } else if (simState.weather === "LIGHT_RAIN") {
    simState.weatherFriction = 0.65;
    rainOverlay.classList.add("active-light");
    logEvent("weather", "SP3: Weather transition -> LIGHT RAIN. Wet road (Friction μ: 0.65, 80% speed limit).");
  } else {
    simState.weatherFriction = 0.40;
    rainOverlay.classList.add("active-heavy");
    logEvent("weather", "SP3: Weather transition -> HEAVY RAIN. Slippery road & reduced visibility (Friction μ: 0.40, 50% speed limit).");
  }
});

// Clear DB
document.getElementById("btn-clear-mem").addEventListener("click", () => {
  simState.roadMemory = [];
  updateMemoryList();
  logEvent("system", "Persistent SQLite Road Memory database cleared.");
});

// Pause / Reset
document.getElementById("btn-pause").addEventListener("click", () => {
  simState.isRunning = !simState.isRunning;
  document.getElementById("btn-pause").textContent = simState.isRunning ? "Pause" : "Resume";
});

document.getElementById("btn-reset").addEventListener("click", () => {
  resetVehicle();
  logEvent("system", "Vehicle position reset to Node A.");
});

// Update Road Memory UI
function updateMemoryList() {
  chipMemory.textContent = `ROAD MEMORY: ${simState.roadMemory.length} HAZARDS`;
  if (simState.roadMemory.length === 0) {
    memoryList.innerHTML = '<div class="empty-mem">Database empty. Run Trip 1 to record hazards.</div>';
    return;
  }

  memoryList.innerHTML = "";
  simState.roadMemory.forEach(p => {
    const item = document.createElement("div");
    item.className = "mem-item";
    item.innerHTML = `
      <span>[${p.id.substring(8)}] (${p.x.toFixed(1)}m, ${p.y.toFixed(1)}m)</span>
      <span>Sev: <strong>${p.severity.toFixed(2)}</strong> | Conf: <strong>${p.confidence.toFixed(2)}</strong></span>
    `;
    memoryList.appendChild(item);
  });
}

// 12-Step Decision Cycle Update
function updateSimulation(dt) {
  if (!simState.isRunning) return;

  const v = simState.veh;
  const isMain = simState.activeRoute === "seg_main_arterial";
  const totalLength = isMain ? 200 : 216;

  // Check if reached destination
  if (v.s >= totalLength) {
    v.speedKmh = Math.max(0, v.speedKmh - 12.0 * dt);
    if (v.speedKmh <= 0.5) {
      simState.isRunning = false;
      logEvent("route", `GOAL REACHED: Trip completed at Node B. Final avg speed maintained.`);
      return;
    }
  }

  // --- 1. SENSOR PERCEPTION & POTHOLE DETECTION ---
  let nearestPotholeDist = 999;
  let activePothole = null;

  if (isMain) {
    simState.potholes.forEach(p => {
      const dist = Math.hypot(p.x - v.x, p.y - v.y);
      if (dist <= 25.0 && p.x >= v.x - 1.0) {
        if (!p.detected) {
          p.detected = true;
          // Store in road memory with spatial deduplication
          const existing = simState.roadMemory.find(ep => Math.hypot(ep.x - p.x, ep.y - p.y) <= 2.0);
          if (!existing) {
            simState.roadMemory.push({
              id: p.id,
              x: p.x,
              y: p.y,
              severity: p.severity,
              confidence: 0.95,
            });
            updateMemoryList();
            logEvent("pothole", `SP1 DETECTED: [${p.id}] at x=${p.x.toFixed(1)}m (Severity: ${p.severity.toFixed(2)}). Stored in SQLite.`);
          }
        }
        if (dist < nearestPotholeDist) {
          nearestPotholeDist = dist;
          activePothole = p;
        }
      }
    });
  }

  // --- 2. LOCAL COLLISION AVOIDANCE PATH PLANNING ---
  if (activePothole && nearestPotholeDist < 18.0) {
    v.activeAvoidance = true;
    v.evadingId = activePothole.id;
    // Lateral swerve: move away from pothole's y coordinate
    const targetOffset = activePothole.y >= 0 ? -1.1 : 1.1;
    v.lateralOffsetM += (targetOffset - v.lateralOffsetM) * 4.0 * dt;
    // Pothole safe speed
    simState.arbitration.potholeSafe = Math.max(15.0, 25.0 - 10.0 * activePothole.severity);
  } else {
    v.activeAvoidance = false;
    v.evadingId = null;
    v.lateralOffsetM += (0.0 - v.lateralOffsetM) * 3.0 * dt;
    simState.arbitration.potholeSafe = 40.0;
  }

  // --- 3. CROWD RISK EVALUATION ---
  if (isMain) {
    const dToCrowd = simState.crowdCluster.s - v.s;
    const isCrowdActive = simState.crowdCluster.active;
    const histCrowd = 0.75;
    const currCrowd = isCrowdActive && dToCrowd > -5 && dToCrowd < 35 ? 0.90 : 0.20;

    const combinedCrowd = 0.4 * histCrowd + 0.6 * currCrowd;

    if (dToCrowd > 0 && dToCrowd <= 35.0) {
      const distFactor = Math.min(1.0, Math.max(0.2, 1.0 - (dToCrowd - 8.0) / 27.0));
      simState.arbitration.crowdSafe = Math.max(12.0, 40.0 - (40.0 - 12.0) * combinedCrowd * distFactor);
    } else if (dToCrowd <= 0 && dToCrowd >= -15.0 && isCrowdActive) {
      simState.arbitration.crowdSafe = 14.0;
    } else {
      simState.arbitration.crowdSafe = 40.0;
    }
  } else {
    simState.arbitration.crowdSafe = 50.0;
  }

  // --- 4. WEATHER RISK EVALUATION ---
  if (simState.weather === "CLEAR") {
    simState.arbitration.weatherSafe = 40.0;
  } else if (simState.weather === "LIGHT_RAIN") {
    simState.arbitration.weatherSafe = 32.0; // 40 * 0.8
  } else {
    simState.arbitration.weatherSafe = 20.0; // 40 * 0.5
  }

  // --- 5. MULTI-HAZARD SPEED ARBITRATION ---
  const roadLimit = isMain ? 40.0 : 50.0;
  simState.arbitration.roadLimit = roadLimit;
  simState.arbitration.base = 40.0;

  const candidates = {
    pothole_avoidance: simState.arbitration.potholeSafe,
    crowd_density: simState.arbitration.crowdSafe,
    weather_condition: simState.arbitration.weatherSafe,
    road_speed_limit: simState.arbitration.roadLimit,
    base_cruise: simState.arbitration.base,
  };

  let minKey = "base_cruise";
  let minSpeed = 40.0;
  for (const [key, speed] of Object.entries(candidates)) {
    if (speed < minSpeed) {
      minSpeed = speed;
      minKey = key;
    }
  }

  simState.arbitration.activeLimiter = minKey;
  simState.veh.targetSpeedKmh = Math.max(10.0, minSpeed);

  // --- 6. LONGITUDINAL & LATERAL ACTUATION ---
  const targetSpeed = simState.veh.targetSpeedKmh;
  const speedError = targetSpeed - v.speedKmh;
  const maxAccel = 4.0 * simState.weatherFriction; // m/s^2 equivalent
  const maxDecel = 8.0 * simState.weatherFriction;

  if (speedError > 0) {
    v.speedKmh += Math.min(speedError, maxAccel * dt * 3.6);
  } else {
    v.speedKmh -= Math.min(-speedError, maxDecel * dt * 3.6);
  }

  // Kinematic step
  const speedMs = v.speedKmh / 3.6;
  v.s += speedMs * dt;

  if (isMain) {
    v.x = v.s;
    v.y = v.lateralOffsetM;
    v.heading = (v.lateralOffsetM / 1.5) * 0.15;
  } else {
    // Bypass route: A(0,0) -> D(100, 55) -> B(200, 0)
    const leg1Len = Math.hypot(WORLD_CONFIG.bypassApexX, WORLD_CONFIG.bypassApexY);
    if (v.s <= leg1Len) {
      const ratio = v.s / leg1Len;
      v.x = ratio * WORLD_CONFIG.bypassApexX;
      v.y = ratio * WORLD_CONFIG.bypassApexY + v.lateralOffsetM;
      v.heading = Math.atan2(WORLD_CONFIG.bypassApexY, WORLD_CONFIG.bypassApexX);
    } else {
      const s2 = v.s - leg1Len;
      const leg2Len = leg1Len;
      const ratio = Math.min(1.0, s2 / leg2Len);
      v.x = WORLD_CONFIG.bypassApexX + ratio * (200 - WORLD_CONFIG.bypassApexX);
      v.y = WORLD_CONFIG.bypassApexY - ratio * WORLD_CONFIG.bypassApexY + v.lateralOffsetM;
      v.heading = Math.atan2(-WORLD_CONFIG.bypassApexY, 200 - WORLD_CONFIG.bypassApexX);
    }
  }

  // Update dynamic pedestrians
  simState.pedestrians.forEach(p => {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.x < 55 || p.x > 85) p.vx *= -1;
    if (p.y < -7 || p.y > 7) p.vy *= -1;
  });

  updateTelemetryUI();
}

// Update HUD & Arbitration Matrix
function updateTelemetryUI() {
  const v = simState.veh;
  const arb = simState.arbitration;

  txtSpeed.textContent = v.speedKmh.toFixed(1);
  txtTargetSpeed.textContent = v.targetSpeedKmh.toFixed(1);
  speedBar.style.width = `${Math.min(100, (v.speedKmh / 50.0) * 100)}%`;

  valBase.textContent = `${arb.base.toFixed(1)} km/h`;
  valLimit.textContent = `${arb.roadLimit.toFixed(1)} km/h`;
  valPothole.textContent = `${arb.potholeSafe.toFixed(1)} km/h`;
  valCrowd.textContent = `${arb.crowdSafe.toFixed(1)} km/h`;
  valWeather.textContent = `${arb.weatherSafe.toFixed(1)} km/h`;

  // Highlight active limiter row
  [rowBase, rowLimit, rowPothole, rowCrowd, rowWeather].forEach(r => r.classList.remove("active-constraint"));
  if (arb.activeLimiter === "pothole_avoidance") {
    rowPothole.classList.add("active-constraint");
    txtActiveLimiter.textContent = "POTHOLE SWERVE";
    txtActiveLimiter.className = "badge-limiter restricted";
  } else if (arb.activeLimiter === "crowd_density") {
    rowCrowd.classList.add("active-constraint");
    txtActiveLimiter.textContent = "CROWD BRAKING";
    txtActiveLimiter.className = "badge-limiter restricted";
  } else if (arb.activeLimiter === "weather_condition") {
    rowWeather.classList.add("active-constraint");
    txtActiveLimiter.textContent = "WEATHER SLOWDOWN";
    txtActiveLimiter.className = "badge-limiter restricted";
  } else if (arb.activeLimiter === "road_speed_limit") {
    rowLimit.classList.add("active-constraint");
    txtActiveLimiter.textContent = "ROAD LIMIT";
    txtActiveLimiter.className = "badge-limiter";
  } else {
    rowBase.classList.add("active-constraint");
    txtActiveLimiter.textContent = "CRUISING";
    txtActiveLimiter.className = "badge-limiter";
  }
}

// Render Simulation Canvas
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Grid Background
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  // 2. Draw Ring Bypass Route (Node A -> Node D -> Node B)
  const nodeA = worldToCanvas(0, 0);
  const nodeD = worldToCanvas(WORLD_CONFIG.bypassApexX, WORLD_CONFIG.bypassApexY);
  const nodeB = worldToCanvas(200, 0);

  ctx.strokeStyle = simState.activeRoute === "seg_bypass" ? "#00f0ff" : "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(nodeA.cx, nodeA.cy);
  ctx.lineTo(nodeD.cx, nodeD.cy);
  ctx.lineTo(nodeB.cx, nodeB.cy);
  ctx.stroke();

  // Bypass centerline dash
  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Draw Main Arterial Road (Node A -> Node B)
  ctx.strokeStyle = simState.activeRoute === "seg_main_arterial" ? "#38bdf8" : "rgba(100, 116, 139, 0.4)";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(nodeA.cx, nodeA.cy);
  ctx.lineTo(nodeB.cx, nodeB.cy);
  ctx.stroke();

  // Centerline markings
  ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
  ctx.lineWidth = 2;
  ctx.setLineDash([12, 10]);
  ctx.stroke();
  ctx.setLineDash([]);

  // 4. Draw Pedestrian Crowd Zone at s=70m
  const crowdC = worldToCanvas(70, 0);
  if (simState.crowdCluster.active) {
    const grad = ctx.createRadialGradient(crowdC.cx, crowdC.cy, 5, crowdC.cx, crowdC.cy, 60);
    grad.addColorStop(0, "rgba(245, 158, 11, 0.35)");
    grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(crowdC.cx, crowdC.cy, 60, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw pedestrian particles
  simState.pedestrians.forEach(p => {
    const pc = worldToCanvas(p.x, p.y);
    ctx.fillStyle = simState.crowdCluster.active ? "#f59e0b" : "rgba(148, 163, 184, 0.6)";
    ctx.beginPath();
    ctx.arc(pc.cx, pc.cy, p.size, 0, Math.PI * 2);
    ctx.fill();
  });

  // 5. Draw Junction Nodes
  [
    { label: "Node A (Start)", c: nodeA },
    { label: "Node D (Bypass Jct)", c: nodeD },
    { label: "Node B (Goal)", c: nodeB },
  ].forEach(n => {
    ctx.fillStyle = "#0f172a";
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(n.c.cx, n.c.cy, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#f1f5f9";
    ctx.font = "bold 11px Outfit";
    ctx.fillText(n.label, n.c.cx - 25, n.c.cy + 22);
  });

  // 6. Draw Potholes on Main Arterial
  simState.potholes.forEach(p => {
    const pt = worldToCanvas(p.x, p.y);
    const radiusPx = p.radius * WORLD_CONFIG.scale * 1.5;

    // Outer warning ring
    ctx.strokeStyle = p.detected ? "#f43f5e" : "rgba(244, 63, 94, 0.4)";
    ctx.lineWidth = p.detected ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.arc(pt.cx, pt.cy, radiusPx + 4, 0, Math.PI * 2);
    ctx.stroke();

    // Core pothole fill
    ctx.fillStyle = p.detected ? "#f43f5e" : "#881337";
    ctx.beginPath();
    ctx.arc(pt.cx, pt.cy, radiusPx, 0, Math.PI * 2);
    ctx.fill();

    // Label
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.fillText(`P-${p.x.toFixed(0)}m (${(p.severity * 100).toFixed(0)}%)`, pt.cx - 18, pt.cy - 12);
  });

  // 7. Draw Autonomous Vehicle & Forward Sensor Perception Cone
  const v = simState.veh;
  const vPos = worldToCanvas(v.x, v.y);

  // Sensor FOV Arc (90 deg, 25m range)
  const sensorRangePx = 25.0 * WORLD_CONFIG.scale;
  const fovHalfRad = (45 * Math.PI) / 180;

  ctx.save();
  ctx.translate(vPos.cx, vPos.cy);
  ctx.rotate(-v.heading);

  // Sensor cone gradient
  const coneGrad = ctx.createRadialGradient(0, 0, 4, 0, 0, sensorRangePx);
  coneGrad.addColorStop(0, "rgba(0, 240, 255, 0.25)");
  coneGrad.addColorStop(1, "rgba(0, 240, 255, 0.0)");

  ctx.fillStyle = coneGrad;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, sensorRangePx, -fovHalfRad, fovHalfRad);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "rgba(0, 240, 255, 0.5)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Headlight beams
  ctx.fillStyle = "rgba(254, 240, 138, 0.15)";
  ctx.beginPath();
  ctx.moveTo(14, -4);
  ctx.lineTo(60, -18);
  ctx.lineTo(60, 18);
  ctx.lineTo(14, 4);
  ctx.fill();

  // Vehicle Body (4.2m length x 1.8m width)
  const carLenPx = 4.2 * WORLD_CONFIG.scale;
  const carWidPx = 1.8 * WORLD_CONFIG.scale;

  // Car Shadow
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(-carLenPx / 2 + 2, -carWidPx / 2 + 2, carLenPx, carWidPx);

  // Car Chassis
  ctx.fillStyle = "#0284c7";
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-carLenPx / 2, -carWidPx / 2, carLenPx, carWidPx, 4);
  ctx.fill();
  ctx.stroke();

  // Windshield & Roof
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(-carLenPx / 6, -carWidPx / 2 + 2, carLenPx / 2.5, carWidPx - 4);

  // Wheels with pure pursuit steering angle
  ctx.fillStyle = "#1e293b";
  // Rear wheels
  ctx.fillRect(-carLenPx / 2 + 2, -carWidPx / 2 - 2, 6, 3);
  ctx.fillRect(-carLenPx / 2 + 2, carWidPx / 2 - 1, 6, 3);

  // Front steered wheels
  ctx.save();
  ctx.translate(carLenPx / 2 - 6, -carWidPx / 2 - 1);
  ctx.rotate(-v.steerAngleRad);
  ctx.fillRect(-3, -2, 7, 3);
  ctx.restore();

  ctx.save();
  ctx.translate(carLenPx / 2 - 6, carWidPx / 2);
  ctx.rotate(-v.steerAngleRad);
  ctx.fillRect(-3, -1, 7, 3);
  ctx.restore();

  ctx.restore();
}

// Main 60 FPS Animation Loop
let lastTime = performance.now();
function gameLoop(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000.0);
  lastTime = now;

  updateSimulation(dt);
  render();

  requestAnimationFrame(gameLoop);
}

// Start simulation visualizer
requestAnimationFrame(gameLoop);
logEvent("system", "Visual engine running at 60 FPS. Click 'Run Trip 1' to start!");
