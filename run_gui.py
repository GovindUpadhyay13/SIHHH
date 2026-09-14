"""Interactive Native Desktop GUI Simulation for Autonomous Driving on Indian Roads.
Built with Pygame.

Controls:
  [1] - Run Trip 1 (Direct arterial with pothole detection, local swerving, and memory saving)
  [2] - Run Trip 2 (Memory-informed global route re-planning to alternate bypass)
  [C] - Toggle Pedestrian Crowd Surge
  [W] - Cycle Weather (CLEAR -> LIGHT_RAIN -> HEAVY_RAIN)
  [R] - Reset vehicle to Start (Node A)
  [ESC/Q] - Exit
"""

import sys
import math
import random
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
from src.models import WeatherCondition
from src.memory.road_memory import SQLiteRoadMemoryStore
from src.config import ROAD_MEMORY_CONFIG

# Configuration
WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 680
FPS = 60

# Palette
COLOR_BG = (11, 15, 25)
COLOR_PANEL = (19, 27, 46)
COLOR_BORDER = (40, 55, 80)
COLOR_TEXT = (241, 245, 249)
COLOR_MUTED = (148, 163, 184)
COLOR_CYAN = (0, 240, 255)
COLOR_EMERALD = (16, 185, 129)
COLOR_AMBER = (245, 158, 11)
COLOR_ROSE = (244, 63, 94)
COLOR_ROAD_MAIN = (56, 189, 248)
COLOR_ROAD_BYPASS = (0, 200, 240)


def world_to_screen(x, y, scale=4.0, orig_x=90, orig_y=420):
    return int(orig_x + x * scale), int(orig_y - y * scale)


def run_desktop_gui():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("AutoDrive — Autonomous Navigation Simulation (Indian Road Network)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Outfit, Arial", 22, bold=True)
    font_sub = pygame.font.SysFont("Outfit, Arial", 14)
    font_hud = pygame.font.SysFont("Consolas, Courier", 14)
    font_speed = pygame.font.SysFont("Consolas, Courier", 36, bold=True)

    # Road memory store
    memory_store = SQLiteRoadMemoryStore()

    # Simulation State
    sim_trip = 1
    active_route = "seg_main_arterial"
    weather = "CLEAR"
    friction = 0.85
    crowd_surge = False

    # Vehicle Kinematic State
    veh_s = 0.0
    veh_speed_kmh = 10.0
    veh_target_speed_kmh = 40.0
    veh_lateral_offset = 0.0
    veh_active_avoidance = False
    active_limiter = "CRUISING"

    # Ground truth potholes
    potholes = [
        {"id": "pothole_km_035", "x": 35.0, "y": 0.2, "severity": 0.85, "detected": False, "r": 0.7},
        {"id": "pothole_km_085", "x": 85.0, "y": -0.3, "severity": 0.75, "detected": False, "r": 0.6},
        {"id": "pothole_km_140", "x": 140.0, "y": 0.1, "severity": 0.90, "detected": False, "r": 0.8},
    ]

    # Pedestrians
    peds = [
        {"x": 60 + random.uniform(0, 25), "y": random.uniform(-4, 4), "vx": random.uniform(-0.3, 0.3), "vy": random.uniform(-0.3, 0.3)}
        for _ in range(25)
    ]

    # Rain particles
    rain_drops = [
        [random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT), random.randint(6, 12)]
        for _ in range(120)
    ]

    log_messages = ["[SYSTEM] Native Pygame simulation visualizer initialized."]

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_1:
                    sim_trip = 1
                    active_route = "seg_main_arterial"
                    veh_s = 0.0
                    veh_speed_kmh = 10.0
                    veh_lateral_offset = 0.0
                    for p in potholes:
                        p["detected"] = False
                    log_messages.append("[TRIP 1] Started on main arterial road. Detecting potholes...")
                elif event.key == pygame.K_2:
                    sim_trip = 2
                    veh_s = 0.0
                    veh_speed_kmh = 10.0
                    veh_lateral_offset = 0.0
                    risk = memory_store.calculate_segment_risk("seg_main_arterial")
                    if risk >= 1.2 or len(memory_store.get_all_potholes()) >= 2:
                        active_route = "seg_bypass"
                        log_messages.append(f"[TRIP 2] Risk={risk:.2f} >= 1.20! Recommending alternate bypass route.")
                    else:
                        active_route = "seg_main_arterial"
                        log_messages.append(f"[TRIP 2] Arterial risk low ({risk:.2f}). Taking main road.")
                elif event.key == pygame.K_c:
                    crowd_surge = not crowd_surge
                    status = "ACTIVE" if crowd_surge else "CLEARED"
                    log_messages.append(f"[CROWD] Dynamic pedestrian surge {status} at s=70m.")
                elif event.key == pygame.K_w:
                    if weather == "CLEAR":
                        weather = "LIGHT_RAIN"
                        friction = 0.65
                    elif weather == "LIGHT_RAIN":
                        weather = "HEAVY_RAIN"
                        friction = 0.40
                    else:
                        weather = "CLEAR"
                        friction = 0.85
                    log_messages.append(f"[WEATHER] Changed to {weather} (Friction: {friction:.2f}).")
                elif event.key == pygame.K_r:
                    veh_s = 0.0
                    veh_speed_kmh = 10.0
                    veh_lateral_offset = 0.0
                    log_messages.append("[RESET] Vehicle returned to Node A.")

        # Kinematic Updates
        is_main = (active_route == "seg_main_arterial")
        total_len = 200.0 if is_main else 216.0

        # Calculate coordinates
        if is_main:
            veh_x = veh_s
            veh_y = veh_lateral_offset
            veh_heading = 0.0
        else:
            leg1 = math.hypot(100.0, 55.0)
            if veh_s <= leg1:
                ratio = veh_s / leg1
                veh_x = ratio * 100.0
                veh_y = ratio * 55.0 + veh_lateral_offset
                veh_heading = math.atan2(55.0, 100.0)
            else:
                s2 = min(leg1, veh_s - leg1)
                ratio = s2 / leg1
                veh_x = 100.0 + ratio * 100.0
                veh_y = 55.0 - ratio * 55.0 + veh_lateral_offset
                veh_heading = math.atan2(-55.0, 100.0)

        # 1. Pothole Detection & Avoidance
        pothole_safe_speed = 40.0
        nearest_p_dist = 999.0
        active_p = None

        if is_main:
            for p in potholes:
                dist = math.hypot(p["x"] - veh_x, p["y"] - veh_y)
                if dist <= 25.0 and p["x"] >= veh_x - 1.0:
                    if not p["detected"]:
                        p["detected"] = True
                        log_messages.append(f"[PERCEPTION] Pothole [{p['id']}] detected at {p['x']:.1f}m!")
                    if dist < nearest_p_dist:
                        nearest_p_dist = dist
                        active_p = p

        if active_p and nearest_p_dist < 18.0:
            veh_active_avoidance = True
            target_offset = -1.1 if active_p["y"] >= 0 else 1.1
            veh_lateral_offset += (target_offset - veh_lateral_offset) * 4.0 * dt
            pothole_safe_speed = max(15.0, 25.0 - 10.0 * active_p["severity"])
        else:
            veh_active_avoidance = False
            veh_lateral_offset += (0.0 - veh_lateral_offset) * 3.0 * dt

        # 2. Crowd Speed Arbitration
        crowd_safe_speed = 40.0
        if is_main:
            d_crowd = 70.0 - veh_s
            curr_c = 0.90 if crowd_surge and -5 < d_crowd < 35 else 0.20
            comb = 0.4 * 0.75 + 0.6 * curr_c
            if 0 < d_crowd <= 35.0:
                dist_factor = min(1.0, max(0.2, 1.0 - (d_crowd - 8.0) / 27.0))
                crowd_safe_speed = max(12.0, 40.0 - (40.0 - 12.0) * comb * dist_factor)
            elif -15.0 <= d_crowd <= 0 and crowd_surge:
                crowd_safe_speed = 14.0

        # 3. Weather Safe Speed
        if weather == "CLEAR":
            weather_safe_speed = 40.0
        elif weather == "LIGHT_RAIN":
            weather_safe_speed = 32.0
        else:
            weather_safe_speed = 20.0

        # 4. Arbitration
        road_limit = 40.0 if is_main else 50.0
        target = min(road_limit, 40.0, pothole_safe_speed, crowd_safe_speed, weather_safe_speed)
        target = max(10.0, target)
        veh_target_speed_kmh = target

        if pothole_safe_speed == target and pothole_safe_speed < 40.0:
            active_limiter = "POTHOLE SWERVE"
        elif crowd_safe_speed == target and crowd_safe_speed < 40.0:
            active_limiter = "CROWD BRAKING"
        elif weather_safe_speed == target and weather_safe_speed < 40.0:
            active_limiter = "WEATHER REDUCTION"
        else:
            active_limiter = "CRUISING"

        # Actuation
        sp_diff = veh_target_speed_kmh - veh_speed_kmh
        if sp_diff > 0:
            veh_speed_kmh += min(sp_diff, 12.0 * friction * dt)
        else:
            veh_speed_kmh -= min(-sp_diff, 20.0 * friction * dt)

        if veh_s < total_len:
            veh_s += (veh_speed_kmh / 3.6) * dt

        # Update pedestrians
        for ped in peds:
            ped["x"] += ped["vx"] * dt
            ped["y"] += ped["vy"] * dt
            if ped["x"] < 55 or ped["x"] > 85: ped["vx"] *= -1
            if ped["y"] < -5 or ped["y"] > 5: ped["vy"] *= -1

        # --- RENDERING ---
        screen.fill(COLOR_BG)

        # Header Bar
        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, WINDOW_WIDTH, 56))
        pygame.draw.line(screen, COLOR_BORDER, (0, 56), (WINDOW_WIDTH, 56), 1)

        title_surf = font_title.render("AutoDrive Simulation", True, COLOR_CYAN)
        screen.blit(title_surf, (24, 8))
        sub_surf = font_sub.render("Adaptive Path Planning & Collision Avoidance on Unstructured Indian Roads", True, COLOR_MUTED)
        screen.blit(sub_surf, (24, 32))

        # Status Badges
        mode_badge = font_hud.render(f"[MODE: TRIP {sim_trip}]", True, COLOR_EMERALD)
        screen.blit(mode_badge, (700, 18))
        weather_badge = font_hud.render(f"[WEATHER: {weather}]", True, COLOR_AMBER if "RAIN" in weather else COLOR_CYAN)
        screen.blit(weather_badge, (860, 18))

        # Roads Rendering
        node_a = world_to_screen(0, 0)
        node_d = world_to_screen(100, 55)
        node_b = world_to_screen(200, 0)

        # Draw Bypass Route
        bypass_color = COLOR_CYAN if active_route == "seg_bypass" else (40, 70, 100)
        pygame.draw.lines(screen, bypass_color, False, [node_a, node_d, node_b], 12)
        # Bypass dashed line
        pygame.draw.lines(screen, (200, 220, 240), False, [node_a, node_d, node_b], 2)

        # Draw Main Road
        main_color = COLOR_ROAD_MAIN if is_main else (50, 65, 90)
        pygame.draw.line(screen, main_color, node_a, node_b, 18)
        # Center markings
        pygame.draw.line(screen, COLOR_AMBER, node_a, node_b, 2)

        # Draw Pedestrians
        for ped in peds:
            px, py = world_to_screen(ped["x"], ped["y"])
            p_color = COLOR_AMBER if crowd_surge else (120, 140, 170)
            pygame.draw.circle(screen, p_color, (px, py), 3)

        if crowd_surge:
            cx, cy = world_to_screen(70, 0)
            s_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
            pygame.draw.circle(s_surf, (245, 158, 11, 45), (60, 60), 50)
            screen.blit(s_surf, (cx - 60, cy - 60))

        # Draw Junction Nodes
        for name, pt in [("Node A (Start)", node_a), ("Node D (Bypass Jct)", node_d), ("Node B (Goal)", node_b)]:
            pygame.draw.circle(screen, (15, 23, 42), pt, 9)
            pygame.draw.circle(screen, COLOR_CYAN, pt, 9, 3)
            lbl = font_hud.render(name, True, COLOR_TEXT)
            screen.blit(lbl, (pt[0] - 30, pt[1] + 16))

        # Draw Potholes
        for p in potholes:
            px, py = world_to_screen(p["x"], p["y"])
            r_px = int(p["r"] * 4.0 * 1.5)
            p_color = COLOR_ROSE if p["detected"] else (136, 19, 55)
            pygame.draw.circle(screen, p_color, (px, py), r_px)
            pygame.draw.circle(screen, COLOR_ROSE, (px, py), r_px + 4, 2 if p["detected"] else 1)
            p_lbl = font_hud.render(f"P-{int(p['x'])}m", True, COLOR_MUTED)
            screen.blit(p_lbl, (px - 16, py - 20))

        # Draw Autonomous Vehicle
        vx_px, vy_px = world_to_screen(veh_x, veh_y)

        # Sensor cone
        cone_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
        pygame.draw.polygon(
            cone_surf,
            (0, 240, 255, 40),
            [(100, 100), (190, 65), (190, 135)],
        )
        rotated_cone = pygame.transform.rotate(cone_surf, math.degrees(veh_heading))
        rc_rect = rotated_cone.get_rect(center=(vx_px, vy_px))
        screen.blit(rotated_cone, rc_rect)

        # Vehicle chassis (rotated)
        car_surf = pygame.Surface((22, 12), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, (2, 132, 199), (0, 0, 22, 12), border_radius=3)
        pygame.draw.rect(car_surf, (56, 189, 248), (0, 0, 22, 12), 2, border_radius=3)
        pygame.draw.rect(car_surf, (15, 23, 42), (4, 2, 10, 8))  # Windshield
        rot_car = pygame.transform.rotate(car_surf, math.degrees(veh_heading))
        rc_car_rect = rot_car.get_rect(center=(vx_px, vy_px))
        screen.blit(rot_car, rc_car_rect)

        # Rain animation
        if "RAIN" in weather:
            rain_color = (147, 197, 253, 160 if weather == "HEAVY_RAIN" else 80)
            for drop in rain_drops:
                drop[0] -= 2
                drop[1] += drop[2]
                if drop[1] > WINDOW_HEIGHT:
                    drop[1] = 0
                    drop[0] = random.randint(0, WINDOW_WIDTH)
                pygame.draw.line(screen, (147, 197, 253), (drop[0], drop[1]), (drop[0] - 2, drop[1] + drop[2]), 1)

        # Telemetry HUD Card (Bottom Right)
        pygame.draw.rect(screen, COLOR_PANEL, (WINDOW_WIDTH - 360, WINDOW_HEIGHT - 210, 340, 190), border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, (WINDOW_WIDTH - 360, WINDOW_HEIGHT - 210, 340, 190), 1, border_radius=8)

        hud_title = font_sub.render("TELEMETRY & SPEED ARBITRATION", True, COLOR_CYAN)
        screen.blit(hud_title, (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 200))

        sp_text = font_speed.render(f"{veh_speed_kmh:.1f}", True, COLOR_TEXT)
        screen.blit(sp_text, (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 170))
        kmh_text = font_hud.render("KM/H", True, COLOR_MUTED)
        screen.blit(kmh_text, (WINDOW_WIDTH - 250, WINDOW_HEIGHT - 155))

        target_lbl = font_hud.render(f"Target: {veh_target_speed_kmh:.1f} km/h", True, COLOR_TEXT)
        screen.blit(target_lbl, (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 128))

        limiter_color = COLOR_ROSE if active_limiter != "CRUISING" else COLOR_EMERALD
        limiter_lbl = font_hud.render(f"Limiter: [{active_limiter}]", True, limiter_color)
        screen.blit(limiter_lbl, (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 106))

        # Speed bar
        bar_w = int(min(1.0, veh_speed_kmh / 50.0) * 310)
        pygame.draw.rect(screen, (30, 45, 70), (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 80, 310, 8), border_radius=4)
        pygame.draw.rect(screen, COLOR_CYAN, (WINDOW_WIDTH - 345, WINDOW_HEIGHT - 80, bar_w, 8), border_radius=4)

        # Controls Hint (Bottom Left)
        ctrl_surf = font_hud.render(
            "[1] Trip 1  |  [2] Trip 2  |  [C] Crowd  |  [W] Weather  |  [R] Reset  |  [ESC] Quit",
            True,
            COLOR_MUTED,
        )
        screen.blit(ctrl_surf, (24, WINDOW_HEIGHT - 35))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_desktop_gui()
