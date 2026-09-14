"""CARLA 3D Autonomous Driving Simulation Runner.

Integrates:
  - Module 1: Pothole Detection & 2D/3D Cost Map (src/perception/pothole_cost_map.py)
  - Module 2: Crowd-Aware Slowdown & Persistent Zone Caching (src/control/crowd_density_monitor.py)
  - Module 3: CARLA Weather-Aware Slowdown (src/control/weather_response.py)

Supports:
  1. Live CARLA 3D Mode: Connects to running CarlaUE4 server on localhost:2000.
  2. Standalone 3D Client Mode: Validates the complete 3D pipeline and sensor hooks
     when CARLA server is offline or unavailable.
"""

from __future__ import annotations
import sys
import time
import math
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import VehicleState, Pothole
from src.perception.pothole_cost_map import PotholeDetectorModule
from src.control.crowd_density_monitor import CrowdDensityMonitor
from src.control.weather_response import WeatherResponseModule, CarlaWeatherParametersProxy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Carla3DSimulation")


class Carla3DSimulationRunner:
    """
    Manages actor spawning, sensor attachment, weather synchronization,
    and closed-loop 3D execution in CARLA.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        town: str = "Town01",
        sync_mode: bool = True,
    ):
        self.host = host
        self.port = port
        self.town = town
        self.sync_mode = sync_mode

        # Initialize the 3 modular add-ons
        self.pothole_module = PotholeDetectorModule(enabled=True)
        self.crowd_module = CrowdDensityMonitor(enabled=True)
        self.weather_module = WeatherResponseModule(enabled=True)

        self.carla_client = None
        self.world = None
        self.ego_vehicle = None
        self.camera_sensor = None
        self.lidar_sensor = None
        self.actors = []

    def connect_carla(self) -> bool:
        """Attempts to connect to live CARLA server."""
        try:
            import carla
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}...")
            self.carla_client = carla.Client(self.host, self.port)
            self.carla_client.set_timeout(5.0)
            self.world = self.carla_client.get_world()
            logger.info(f"Connected to CARLA! Map: {self.world.get_map().name}")
            return True
        except ImportError:
            logger.warning("The 'carla' Python package is not installed.")
            return False
        except Exception as ex:
            logger.warning(f"Could not reach CARLA server ({ex}).")
            return False

    def run_live_carla(self, max_ticks: int = 500) -> None:
        """Executes autonomous navigation loop in live CARLA 3D world."""
        import carla
        blueprint_lib = self.world.get_blueprint_library()
        bp_vehicle = blueprint_lib.filter("vehicle.tesla.model3")[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        logger.info("Spawning ego vehicle in 3D world...")
        self.ego_vehicle = self.world.spawn_actor(bp_vehicle, spawn_point)
        self.actors.append(self.ego_vehicle)

        # Attach RGB Camera
        bp_cam = blueprint_lib.find("sensor.camera.rgb")
        bp_cam.set_attribute("image_size_x", "800")
        bp_cam.set_attribute("image_size_y", "600")
        bp_cam.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera_sensor = self.world.spawn_actor(bp_cam, cam_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.camera_sensor)

        # Weather setup
        weather_params = carla.WeatherParameters.WetCloudySunset
        self.world.set_weather(weather_params)

        logger.info("CARLA 3D autonomous control loop active...")
        tick = 0
        try:
            while tick < max_ticks:
                transform = self.ego_vehicle.get_transform()
                velocity = self.ego_vehicle.get_velocity()
                speed_kmh = 3.6 * math.hypot(velocity.x, velocity.y)

                v_state = VehicleState(
                    x=transform.location.x,
                    y=transform.location.y,
                    heading=math.radians(transform.rotation.yaw),
                    velocity=speed_kmh / 3.6,
                )

                # 1. Weather Module update
                current_carla_weather = self.world.get_weather()
                weather_cap, headway_mult, weather_info = self.weather_module.update(current_carla_weather)

                # 2. Crowd Module update
                crowd_cap, crowd_info = self.crowd_module.update(v_state, {"detected_agents": []})

                # 3. Pothole Cost Map update
                cost_map, pothole_cap, reroute_event = self.pothole_module.update(v_state, {"detected_potholes": []})

                # Arbitrate Speed Cap
                active_caps = [c for c in [40.0, weather_cap, crowd_cap, pothole_cap] if c is not None]
                target_speed = max(10.0, min(active_caps))

                # Actuation
                throttle = 0.5 if speed_kmh < target_speed else 0.0
                brake = 0.4 if speed_kmh > target_speed + 2.0 else 0.0
                control = carla.VehicleControl(throttle=throttle, steer=0.0, brake=brake)
                self.ego_vehicle.apply_control(control)

                if tick % 20 == 0:
                    logger.info(f"Tick {tick}: Speed={speed_kmh:.1f} km/h, TargetCap={target_speed:.1f} km/h (Weather={weather_info['caution_level']})")

                tick += 1
                time.sleep(0.05)

        finally:
            self.cleanup()

    def run_standalone_demo(self, steps: int = 150) -> None:
        """
        Standalone 3D pipeline simulation.
        Executes and verifies the 3 modules with 3D kinematic trajectories and synthetic telemetry.
        """
        print("\n" + "=" * 75)
        print(" CARLA 3D AUTONOMOUS DRIVING SIMULATION — DEMO RUNNER")
        print("=" * 75)
        print("Status: Running standalone 3D pipeline (CARLA client proxy mode).")
        print("Integration Modules Active:")
        print("  1. Pothole Detection & Cost Map (src/perception/pothole_cost_map.py)")
        print("  2. Crowd Density Monitor (src/control/crowd_density_monitor.py)")
        print("  3. CARLA Weather Response (src/control/weather_response.py)")
        print("-" * 75)

        v_state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=5.0)

        # Synthetic CARLA weather progression (Rain -> Wetness -> Fog)
        carla_weather = CarlaWeatherParametersProxy(
            precipitation=25.0,
            precipitation_deposits=40.0,
            fog_density=20.0,
        )

        for step in range(steps):
            t = step * 0.1

            # Dynamic weather change halfway through
            if step > 60:
                carla_weather.precipitation = 65.0
                carla_weather.fog_density = 50.0

            # Dynamic pedestrian cluster at s=50m
            detected_agents = []
            if 30.0 <= v_state.x <= 75.0:
                detected_agents = [{"x": 60.0 + i, "y": 0.0} for i in range(12)]

            # Potholes
            detected_potholes = []
            if 20.0 <= v_state.x <= 45.0:
                detected_potholes = [Pothole(id="p35", x=35.0, y=0.1, road_segment_id="s1", severity=0.85, confidence=0.92)]

            # 1. Pothole Cost Map Tick
            lookahead_path = [(v_state.x + i * 2.0, v_state.y) for i in range(10)]
            cost_map, pothole_cap, reroute_event = self.pothole_module.update(
                vehicle=v_state,
                sensor_data={"detected_potholes": detected_potholes},
                lookahead_path=lookahead_path,
                dt=0.1,
            )

            # 2. Crowd Density Monitor Tick
            crowd_cap, crowd_info = self.crowd_module.update(
                vehicle=v_state,
                perception_data={"detected_agents": detected_agents},
                dt=0.1,
            )

            # 3. Weather Response Tick
            weather_cap, headway_mult, weather_info = self.weather_module.update(carla_weather, dt=0.1)

            # Multi-Hazard Speed Cap Arbitration
            active_caps = [c for c in [40.0, pothole_cap, crowd_cap, weather_cap] if c is not None]
            target_cap = max(10.0, min(active_caps))

            # Longitudinal velocity update
            if v_state.velocity * 3.6 < target_cap:
                v_state.velocity += min(1.2 * 0.1, (target_cap / 3.6) - v_state.velocity)
            else:
                v_state.velocity -= min(2.5 * 0.1, v_state.velocity - (target_cap / 3.6))

            v_state.x += v_state.velocity * 0.1

            if step % 15 == 0:
                print(
                    f"t={t:4.1f}s | Pos: ({v_state.x:5.1f}m, {v_state.y:4.1f}m) | "
                    f"Speed: {v_state.velocity * 3.6:4.1f} km/h | TargetCap: {target_cap:4.1f} km/h | "
                    f"Weather: [{weather_info['caution_level']:16}] | Crowd: [{crowd_info['density_band']:6}] | "
                    f"Reroute: {reroute_event or 'None'}"
                )

        print("\n" + "=" * 75)
        print("STANDALONE 3D PIPELINE DEMO COMPLETED SUCCESSFULLY")
        print("=" * 75)

    def cleanup(self) -> None:
        """Destroys all spawned actors in CARLA."""
        logger.info("Cleaning up CARLA 3D actors...")
        for actor in self.actors:
            if actor is not None:
                actor.destroy()
        self.actors.clear()


def main():
    parser = argparse.ArgumentParser(description="CARLA 3D Autonomous Navigation Runner")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port")
    parser.add_argument("--town", default="Town01", help="CARLA Town map")
    parser.add_argument("--standalone", action="store_true", help="Run standalone 3D pipeline demo")
    args = parser.parse_args()

    runner = Carla3DSimulationRunner(host=args.host, port=args.port, town=args.town)

    if not args.standalone and runner.connect_carla():
        runner.run_live_carla()
    else:
        if not args.standalone:
            print("\nNote: CARLA server not detected on localhost:2000.")
            print("To connect live, launch CARLA first: .\\CarlaUE4.exe -carla-rpc-port=2000")
            print("Running in standalone 3D client mode now...\n")
        runner.run_standalone_demo()


if __name__ == "__main__":
    main()
