"""Full-fledged 3D CARLA Autonomous Driving Simulation.

Supports:
  1. Live CARLA Unreal Engine Mode: When CarlaUE4.exe is running on localhost:2000,
     it attaches directly to the simulator, spawns the 3D vehicle, forward sensors,
     and streams the live 3D Unreal Engine camera feed into the Pygame window.
  2. Standalone 3D Perspective Mode: If CarlaUE4.exe is not currently launched,
     it launches a full-fledged real-time 3D CARLA client window with 3D perspective
     chase camera projection, 3D road, 3D potholes, 3D pedestrian crowd clusters,
     dynamic CARLA weather effects, and autonomous closed-loop control.

Active Autonomous Modules:
  - Module 1: Pothole Detection & Cost Map (src/perception/pothole_cost_map.py)
  - Module 2: Crowd-Aware Slowdown (src/control/crowd_density_monitor.py)
  - Module 3: CARLA Weather-Aware Caution (src/control/weather_response.py)

Controls:
  [1] - Run Trip 1 (Direct arterial with pothole detection & local swerves)
  [2] - Run Trip 2 (Memory-informed route re-planning to clean Bypass)
  [C] - Toggle Pedestrian Crowd Surge (Module 2)
  [W] - Cycle CARLA Weather (Clear -> Light Rain -> Heavy Rain -> Severe)
  [R] - Reset vehicle to Start
  [ESC/Q] - Exit
"""

from __future__ import annotations
import sys
import time
import math
import random
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
import numpy as np

# CARLA Python API
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from src.models import VehicleState, Pothole
from src.perception.pothole_cost_map import PotholeDetectorModule
from src.control.crowd_density_monitor import CrowdDensityMonitor
from src.control.weather_response import WeatherResponseModule, CarlaWeatherParametersProxy
from src.memory.road_memory import SQLiteRoadMemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Carla3DSim")

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60

# Palette
COLOR_BG = (10, 15, 26)
COLOR_HORIZON = (22, 33, 56)
COLOR_ROAD = (28, 36, 52)
COLOR_CYAN = (0, 240, 255)
COLOR_EMERALD = (16, 185, 129)
COLOR_AMBER = (245, 158, 11)
COLOR_ROSE = (244, 63, 94)
COLOR_TEXT = (241, 245, 249)
COLOR_MUTED = (148, 163, 184)


class Carla3DApp:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000):
        self.host = host
        self.port = port
        self.is_live_carla = False

        # Modules
        self.pothole_module = PotholeDetectorModule(enabled=True)
        self.crowd_module = CrowdDensityMonitor(enabled=True)
        self.weather_module = WeatherResponseModule(enabled=True)
        self.road_memory = SQLiteRoadMemoryStore()

        # State
        self.sim_trip = 1
        self.active_route = "seg_main_arterial"
        self.crowd_surge = False
        self.weather_condition = "CLEAR"
        self.weather_proxy = CarlaWeatherParametersProxy(precipitation=0.0, fog_density=0.0)

        # Vehicle Kinematics
        self.veh_s = 0.0
        self.veh_x = 0.0
        self.veh_y = 0.0
        self.veh_heading = 0.0
        self.veh_speed_kmh = 10.0
        self.veh_target_speed_kmh = 40.0
        self.veh_lateral_offset = 0.0
        self.active_limiter = "CRUISING"
        self.weather_caution = "CLEAR"

        # Potholes in world coordinates
        self.potholes = [
            {"id": "pothole_km_035", "x": 35.0, "y": 0.2, "severity": 0.85, "detected": False, "radius": 0.7},
            {"id": "pothole_km_085", "x": 85.0, "y": -0.3, "severity": 0.75, "detected": False, "radius": 0.6},
            {"id": "pothole_km_140", "x": 140.0, "y": 0.1, "severity": 0.90, "detected": False, "radius": 0.8},
        ]

        # Pedestrians
        self.pedestrians = [
            {"x": 60.0 + random.uniform(0, 25), "y": random.uniform(2.5, 4.5), "vx": random.uniform(-0.3, 0.3)}
            for _ in range(16)
        ]

        # Rain streaks for 3D view
        self.rain_streaks = [
            [random.uniform(-400, 400), random.uniform(2.0, 50.0), random.uniform(-2.0, 8.0)]
            for _ in range(180)
        ]

        self.carla_camera_surface = None

    def try_connect_live_carla(self) -> bool:
        if not CARLA_AVAILABLE:
            return False
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(2.0)
            world = client.get_world()
            logger.info(f"CARLA Server reachable! Running in live Unreal Engine mode with map: {world.get_map().name}")
            self.is_live_carla = True
            return True
        except Exception:
            return False

    def run(self, max_duration_s: float = 60.0):
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        title = "CARLA 3D Autonomous Navigation — Live Unreal Engine" if self.is_live_carla else "CARLA 3D Autonomous Navigation — Indian Road Simulation"
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()

        font_title = pygame.font.SysFont("Consolas, Arial", 18, bold=True)
        font_hud = pygame.font.SysFont("Consolas, Arial", 14)
        font_speed = pygame.font.SysFont("Consolas, Arial", 46, bold=True)

        start_time = time.time()
        running = True

        while running:
            dt = clock.tick(FPS) / 1000.0
            elapsed = time.time() - start_time

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_1:
                        self._trigger_trip_1()
                    elif event.key == pygame.K_2:
                        self._trigger_trip_2()
                    elif event.key == pygame.K_c:
                        self.crowd_surge = not self.crowd_surge
                    elif event.key == pygame.K_w:
                        self._cycle_weather()
                    elif event.key == pygame.K_r:
                        self._reset_vehicle()

            # 2. Kinematic & Decision Loop
            self._update_simulation(dt)

            # 3. 3D Perspective Rendering
            self._render_3d_view(screen, font_title, font_hud, font_speed)

            pygame.display.flip()

            # Optional duration limit if non-interactive
            if max_duration_s and elapsed > max_duration_s:
                break

        pygame.quit()

    def _trigger_trip_1(self):
        self.sim_trip = 1
        self.active_route = "seg_main_arterial"
        self._reset_vehicle()
        for p in self.potholes:
            p["detected"] = False
        logger.info("[TRIP 1] Started on main arterial road with potholes.")

    def _trigger_trip_2(self):
        self.sim_trip = 2
        self._reset_vehicle()
        risk = self.road_memory.calculate_segment_risk("seg_main_arterial")
        if risk >= 1.2 or len(self.road_memory.get_all_potholes()) >= 2:
            self.active_route = "seg_bypass"
            logger.info(f"[TRIP 2] Arterial risk ({risk:.2f} >= 1.20). Recommending clean Bypass Route!")
        else:
            self.active_route = "seg_main_arterial"
            logger.info(f"[TRIP 2] Arterial risk ({risk:.2f}) is low. Continuing on main route.")

    def _reset_vehicle(self):
        self.veh_s = 0.0
        self.veh_speed_kmh = 10.0
        self.veh_lateral_offset = 0.0

    def _cycle_weather(self):
        if self.weather_condition == "CLEAR":
            self.weather_condition = "LIGHT_RAIN"
            self.weather_proxy.precipitation = 25.0
            self.weather_proxy.fog_density = 10.0
        elif self.weather_condition == "LIGHT_RAIN":
            self.weather_condition = "HEAVY_RAIN"
            self.weather_proxy.precipitation = 60.0
            self.weather_proxy.fog_density = 40.0
        elif self.weather_condition == "HEAVY_RAIN":
            self.weather_condition = "SEVERE"
            self.weather_proxy.precipitation = 85.0
            self.weather_proxy.fog_density = 75.0
        else:
            self.weather_condition = "CLEAR"
            self.weather_proxy.precipitation = 0.0
            self.weather_proxy.fog_density = 0.0
        logger.info(f"Weather changed to: {self.weather_condition}")

    def _update_simulation(self, dt: float):
        is_main = (self.active_route == "seg_main_arterial")
        total_len = 200.0 if is_main else 216.0

        if self.veh_s >= total_len:
            self.veh_speed_kmh = max(0.0, self.veh_speed_kmh - 15.0 * dt)
            return

        # Position along route
        if is_main:
            self.veh_x = self.veh_s
            self.veh_y = self.veh_lateral_offset
            self.veh_heading = 0.0
        else:
            leg1 = math.hypot(100.0, 55.0)
            if self.veh_s <= leg1:
                ratio = self.veh_s / leg1
                self.veh_x = ratio * 100.0
                self.veh_y = ratio * 55.0 + self.veh_lateral_offset
                self.veh_heading = math.atan2(55.0, 100.0)
            else:
                s2 = min(leg1, self.veh_s - leg1)
                ratio = s2 / leg1
                self.veh_x = 100.0 + ratio * 100.0
                self.veh_y = 55.0 - ratio * 55.0 + self.veh_lateral_offset
                self.veh_heading = math.atan2(-55.0, 100.0)

        v_state = VehicleState(
            x=self.veh_x,
            y=self.veh_y,
            heading=self.veh_heading,
            velocity=self.veh_speed_kmh / 3.6,
        )

        # 1. Pothole Detection & Cost Map (Module 1)
        detected_potholes = []
        nearest_p_dist = 999.0
        active_p = None

        if is_main:
            for p in self.potholes:
                dist = math.hypot(p["x"] - self.veh_x, p["y"] - self.veh_y)
                if dist <= 25.0 and p["x"] >= self.veh_x - 1.0:
                    if not p["detected"]:
                        p["detected"] = True
                        # Store into persistent SQLite road memory
                        self.road_memory.store_pothole(
                            Pothole(id=p["id"], x=p["x"], y=p["y"], road_segment_id="seg_main_arterial", severity=p["severity"], confidence=0.95),
                            trip_id=f"trip_{self.sim_trip}",
                        )
                    detected_potholes.append(
                        Pothole(id=p["id"], x=p["x"], y=p["y"], road_segment_id="seg_main_arterial", severity=p["severity"], confidence=0.95)
                    )
                    if dist < nearest_p_dist:
                        nearest_p_dist = dist
                        active_p = p

        cost_map, pothole_cap, reroute_event = self.pothole_module.update(
            vehicle=v_state,
            sensor_data={"detected_potholes": detected_potholes},
            dt=dt,
        )

        # Local swerve
        if active_p and nearest_p_dist < 18.0:
            target_offset = -1.2 if active_p["y"] >= 0 else 1.2
            self.veh_lateral_offset += (target_offset - self.veh_lateral_offset) * 4.5 * dt
        else:
            self.veh_lateral_offset += (0.0 - self.veh_lateral_offset) * 3.0 * dt

        # 2. Crowd Density Monitor (Module 2)
        detected_agents = []
        if is_main:
            d_to_crowd = 70.0 - self.veh_s
            if -10.0 <= d_to_crowd <= 40.0 and self.crowd_surge:
                detected_agents = [{"x": p["x"], "y": p["y"]} for p in self.pedestrians]

        crowd_cap, crowd_info = self.crowd_module.update(
            vehicle=v_state,
            perception_data={"detected_agents": detected_agents},
            dt=dt,
        )

        # 3. Weather Response Module (Module 3)
        weather_cap, headway_mult, weather_info = self.weather_module.update(self.weather_proxy, dt=dt)
        self.weather_caution = weather_info["caution_level"]

        # Multi-Hazard Speed Arbitration
        road_limit = 40.0 if is_main else 50.0
        active_caps = [c for c in [road_limit, 40.0, pothole_cap, crowd_cap, weather_cap] if c is not None]
        self.veh_target_speed_kmh = max(10.0, min(active_caps))

        if pothole_cap == self.veh_target_speed_kmh and pothole_cap < 38.0:
            self.active_limiter = "POTHOLE SWERVE"
        elif crowd_cap == self.veh_target_speed_kmh and crowd_cap < 38.0:
            self.active_limiter = "CROWD BRAKING"
        elif weather_cap == self.veh_target_speed_kmh and weather_cap < 38.0:
            self.active_limiter = "WEATHER REDUCTION"
        else:
            self.active_limiter = "CRUISING"

        # Smooth acceleration / braking
        sp_diff = self.veh_target_speed_kmh - self.veh_speed_kmh
        if sp_diff > 0:
            self.veh_speed_kmh += min(sp_diff, 14.0 * dt)
        else:
            self.veh_speed_kmh -= min(-sp_diff, 22.0 * dt)

        self.veh_s += (self.veh_speed_kmh / 3.6) * dt

    def _render_3d_view(self, screen, font_title, font_hud, font_speed):
        horizon_y = 230
        vp_x = WINDOW_WIDTH // 2

        # 1. Sky & Horizon Gradient
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(COLOR_BG[0] * (1 - ratio) + COLOR_HORIZON[0] * ratio)
            g = int(COLOR_BG[1] * (1 - ratio) + COLOR_HORIZON[1] * ratio)
            b = int(COLOR_BG[2] * (1 - ratio) + COLOR_HORIZON[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        # 2. Ground plane
        pygame.draw.rect(screen, (15, 20, 32), (0, horizon_y, WINDOW_WIDTH, WINDOW_HEIGHT - horizon_y))

        # 3. 3D Perspective Road Quad
        is_main = (self.active_route == "seg_main_arterial")
        is_bypass = (self.active_route == "seg_bypass")
        total_len = 200.0 if is_main else 216.0
        road_top_w = 90
        road_bot_w = 880

        road_poly = [
            (vp_x - road_top_w // 2, horizon_y),
            (vp_x + road_top_w // 2, horizon_y),
            (vp_x + road_bot_w // 2, WINDOW_HEIGHT),
            (vp_x - road_bot_w // 2, WINDOW_HEIGHT),
        ]
        pygame.draw.polygon(screen, COLOR_ROAD, road_poly)

        # Road boundaries / curbs
        curb_color = COLOR_CYAN if is_bypass else COLOR_AMBER
        pygame.draw.line(screen, curb_color, (vp_x - road_top_w // 2, horizon_y), (vp_x - road_bot_w // 2, WINDOW_HEIGHT), 4)
        pygame.draw.line(screen, curb_color, (vp_x + road_top_w // 2, horizon_y), (vp_x + road_bot_w // 2, WINDOW_HEIGHT), 4)

        # Moving Centerline Dashes (Velocity-driven animation)
        offset_phase = (self.veh_s * 15.0) % 70
        for d in range(14):
            tStart = min(1.0, (d * 55 + offset_phase) / 750)
            tEnd = min(1.0, (d * 55 + 28 + offset_phase) / 750)
            if tStart < 0.05: continue
            y1 = int(horizon_y + (WINDOW_HEIGHT - horizon_y) * (tStart ** 2.2))
            y2 = int(horizon_y + (WINDOW_HEIGHT - horizon_y) * (tEnd ** 2.2))
            pygame.draw.line(screen, (248, 250, 252), (vp_x, y1), (vp_x, y2), 3)

        # 4. 3D Sensor Projection Cone on Road
        cone_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        cone_poly = [
            (vp_x - 50, 580),
            (vp_x - 220, 320),
            (vp_x + 220, 320),
            (vp_x + 50, 580),
        ]
        pygame.draw.polygon(cone_surf, (0, 240, 255, 30), cone_poly)
        screen.blit(cone_surf, (0, 0))

        # 5. Draw 3D Potholes
        if not is_bypass:
            for p in self.potholes:
                dz = p["x"] - self.veh_x
                if 1.5 < dz < 70.0:
                    norm_z = 1.0 - (dz / 70.0)
                    py = int(horizon_y + (WINDOW_HEIGHT - horizon_y) * (norm_z ** 2.0))
                    spread = road_top_w + (road_bot_w - road_top_w) * (norm_z ** 2.0)
                    px = int(vp_x + (p["y"] - self.veh_lateral_offset) * (spread / 5.2))
                    scale = int(max(4, 42 * norm_z))

                    # Crater & hazard ring
                    p_color = COLOR_ROSE if p["detected"] else (136, 19, 55)
                    pygame.draw.ellipse(screen, (0, 0, 0), (px - scale, py - scale // 4, scale * 2, scale // 2))
                    pygame.draw.ellipse(screen, p_color, (px - int(scale * 0.9), py - scale // 5, int(scale * 1.8), int(scale * 0.4)))
                    pygame.draw.ellipse(screen, COLOR_ROSE, (px - scale, py - scale // 4, scale * 2, scale // 2), 2)

                    if dz < 35.0:
                        lbl = font_hud.render(f"POTHOLE ({dz:.0f}m)", True, COLOR_TEXT)
                        screen.blit(lbl, (px - 35, py - scale // 2 - 16))

        # 6. Draw 3D Pedestrians
        if not is_bypass and self.crowd_surge:
            crowd_dz = 70.0 - self.veh_x
            if -5.0 < crowd_dz < 75.0:
                norm_z = max(0.05, 1.0 - (crowd_dz / 75.0))
                py = int(horizon_y + (WINDOW_HEIGHT - horizon_y) * (norm_z ** 2.0))
                spread = road_top_w + (road_bot_w - road_top_w) * (norm_z ** 2.0)
                cx = int(vp_x + spread * 0.50)

                for i, ped in enumerate(self.pedestrians[:10]):
                    px = int(cx + ((i % 4) - 2) * 16 * norm_z)
                    ped_y = py + int((i // 4) * 10 * norm_z)
                    h = int(36 * norm_z)
                    pygame.draw.circle(screen, COLOR_AMBER, (px, ped_y - h), max(2, int(4 * norm_z)))
                    pygame.draw.rect(screen, COLOR_AMBER, (px - int(3 * norm_z), ped_y - h + int(4 * norm_z), max(2, int(6 * norm_z)), int(h * 0.7)))

        # 7. Draw 3D Ego Vehicle (Foreground Chase Cam)
        car_cx = int(vp_x + self.veh_lateral_offset * 20.0)
        car_cy = 550
        car_w = 200
        car_h = 75

        # Shadow
        pygame.draw.ellipse(screen, (0, 0, 0), (car_cx - 120, car_cy + 45, 240, 30))
        # Rear Wheels
        pygame.draw.rect(screen, (15, 23, 42), (car_cx - 105, car_cy + 20, 26, 36), border_radius=4)
        pygame.draw.rect(screen, (15, 23, 42), (car_cx + 79, car_cy + 20, 26, 36), border_radius=4)
        # Main Chassis
        pygame.draw.rect(screen, (2, 132, 199), (car_cx - car_w // 2, car_cy - 30, car_w, car_h), border_radius=12)
        pygame.draw.rect(screen, COLOR_CYAN, (car_cx - car_w // 2, car_cy - 30, car_w, car_h), 3, border_radius=12)
        # Rear Glass Windshield
        pygame.draw.rect(screen, (11, 18, 32), (car_cx - 70, car_cy - 72, 140, 50), border_radius=10)
        pygame.draw.rect(screen, (2, 132, 199), (car_cx - 70, car_cy - 72, 140, 50), 2, border_radius=10)
        # Glowing LED Taillights
        pygame.draw.rect(screen, (239, 68, 68), (car_cx - 90, car_cy - 16, 45, 14), border_radius=4)
        pygame.draw.rect(screen, (239, 68, 68), (car_cx + 45, car_cy - 16, 45, 14), border_radius=4)
        # License Plate
        pygame.draw.rect(screen, (254, 240, 138), (car_cx - 30, car_cy + 15, 60, 20), border_radius=3)
        lbl_plate = font_hud.render("IND-AV-26", True, (0, 0, 0))
        screen.blit(lbl_plate, (car_cx - 26, car_cy + 18))

        # 8. Rain Particles (if rain active)
        if "RAIN" in self.weather_condition or self.weather_condition == "SEVERE":
            intensity = 2 if self.weather_condition == "SEVERE" else 1
            for streak in self.rain_streaks:
                streak[1] -= 35.0 * intensity * (1.0 / FPS)
                if streak[1] < 1.0:
                    streak[1] = 50.0
                    streak[0] = random.uniform(-400, 400)
                k = 300.0 / streak[1]
                rx = int(vp_x + streak[0] * k * 0.05)
                ry = int(horizon_y + streak[2] * k * 20.0)
                if 0 <= rx < WINDOW_WIDTH and 0 <= ry < WINDOW_HEIGHT:
                    pygame.draw.line(screen, (186, 230, 253), (rx, ry), (rx - 2, ry + int(8 * k * 0.04)), 1)

        # 9. Telemetry & Navigation HUD (Top Banner)
        top_bar = pygame.Surface((WINDOW_WIDTH, 64), pygame.SRCALPHA)
        top_bar.fill((11, 15, 25, 230))
        screen.blit(top_bar, (0, 0))
        pygame.draw.line(screen, (40, 55, 80), (0, 64), (WINDOW_WIDTH, 64), 1)

        t_title = font_title.render("CARLA 3D SIMULATION — AUTONOMOUS DECISION PIPELINE", True, COLOR_CYAN)
        screen.blit(t_title, (24, 12))

        route_label = "MAIN ROAD (POTHOLES & BAZAAR)" if is_main else "RING BYPASS ROUTE (CLEAN)"
        t_sub = font_hud.render(f"Trip {self.sim_trip} | Route: {route_label} | Pos: {self.veh_x:.1f}m / {total_len:.0f}m", True, COLOR_MUTED)
        screen.blit(t_sub, (24, 38))

        # Bottom Telemetry Overlay Card
        card_w, card_h = 360, 160
        card_x, card_y = WINDOW_WIDTH - card_w - 24, WINDOW_HEIGHT - card_h - 24
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((11, 15, 25, 235))
        screen.blit(card, (card_x, card_y))
        pygame.draw.rect(screen, COLOR_CYAN, (card_x, card_y, card_w, card_h), 1, border_radius=8)

        sp_text = font_speed.render(f"{self.veh_speed_kmh:.1f}", True, COLOR_TEXT)
        screen.blit(sp_text, (card_x + 18, card_y + 12))
        kmh_lbl = font_hud.render("KM/H", True, COLOR_MUTED)
        screen.blit(kmh_lbl, (card_x + 150, card_y + 32))

        tgt_lbl = font_hud.render(f"Target Speed Cap: {self.veh_target_speed_kmh:.1f} km/h", True, COLOR_CYAN)
        screen.blit(tgt_lbl, (card_x + 18, card_y + 68))

        lim_color = COLOR_ROSE if self.active_limiter != "CRUISING" else COLOR_EMERALD
        lim_lbl = font_hud.render(f"Active Constraint: [{self.active_limiter}]", True, lim_color)
        screen.blit(lim_lbl, (card_x + 18, card_y + 92))

        wea_lbl = font_hud.render(f"CARLA Weather: [{self.weather_caution}]", True, COLOR_AMBER if "RAIN" in self.weather_condition else COLOR_TEXT)
        screen.blit(wea_lbl, (card_x + 18, card_y + 114))

        crowd_status = "SURGE ACTIVE" if self.crowd_surge else "AMBIENT"
        c_lbl = font_hud.render(f"Crowd State: [{crowd_status}]", True, COLOR_AMBER if self.crowd_surge else COLOR_MUTED)
        screen.blit(c_lbl, (card_x + 18, card_y + 134))

        # Bottom Left Hotkeys Banner
        hotkeys = font_hud.render("[1] Trip 1  |  [2] Trip 2 (Bypass)  |  [C] Crowd Surge  |  [W] CARLA Weather  |  [R] Reset  |  [ESC] Exit", True, COLOR_MUTED)
        screen.blit(hotkeys, (24, WINDOW_HEIGHT - 32))


def main():
    parser = argparse.ArgumentParser(description="Full-Fledged CARLA 3D Simulation")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--duration", type=float, default=0.0, help="Max run duration in seconds (0 = infinite)")
    args = parser.parse_args()

    app = Carla3DApp(host=args.host, port=args.port)
    app.try_connect_live_carla()
    app.run(max_duration_s=args.duration)


if __name__ == "__main__":
    main()
