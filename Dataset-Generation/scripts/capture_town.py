import contextlib
import csv
import json
import math
import os
import random
import threading
import time
import traceback
from datetime import datetime, timezone

import carla
import numpy as np
from PIL import Image
from rbac_gateway import RBACGateway
from security_engine import SecurityEngine, dataset_key


def _parse_args():
    """Town selection. Defaults to town01 so the script still runs bare."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrated secure CARLA capture for one town"
    )
    parser.add_argument(
        "--town",
        default="town01",
        choices=["town01", "town02", "town03", "town04", "town05"],
        help="which town to capture (default: town01)",
    )
    parser.add_argument(
        "--sets",
        type=int,
        default=2000,
        help="frame sets to capture (default: 2000); use a small number for a test run",
    )
    return parser.parse_args()


_ARGS = _parse_args()
TOWN_NAME = _ARGS.town
CARLA_TOWN = TOWN_NAME.replace("town", "Town")

OUTPUT_PATH = os.path.join(os.getcwd(), "output", TOWN_NAME)

USER_ROLE = "Simulation_Engineer"
NUM_SETS = _ARGS.sets

CAPTURE_INTERVAL = 16  # frames at 20 FPS, one set every 0.8 s

SENSORS_PER_SET = 3
REQUIRED_SENSORS = ["rgb", "depth", "semantic"]
SAVE_SEMANTIC_AS_CITYSCAPES = True

# Ground-truth 2D bounding boxes, one YOLO label file per frame

SAVE_BOUNDING_BOXES = True
BBOX_MAX_DISTANCE_M = 80.0
BBOX_MIN_PIXEL_AREA = 30

CLASS_IDS = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2,
}

TWO_WHEELER_BLUEPRINTS = {
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
}

# Semantic mask class remapping

SAVE_CLASS_MASKS = True
BACKGROUND_CLASS_ID = 3
CLASS_NAMES_WITH_BACKGROUND = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "background"}

CLASS_COLOR_MAP = {
    (220, 20, 60):  CLASS_IDS["pedestrian"],  # Pedestrian
    (255, 0, 0):    CLASS_IDS["cyclist"],     # Rider / bike rider
    (0, 0, 142):    CLASS_IDS["car"],         # Car
    (0, 0, 70):     CLASS_IDS["car"],         # Truck -> grouped as car
    (0, 60, 100):   CLASS_IDS["car"],         # Bus -> grouped as car
    (0, 0, 230):    CLASS_IDS["cyclist"],     # Motorcycle -> grouped as cyclist
    (119, 11, 32):  CLASS_IDS["cyclist"],     # Bicycle
}

_CLASS_LUT = np.full(256 ** 3, BACKGROUND_CLASS_ID, dtype=np.uint8)
for _color, _class_id in CLASS_COLOR_MAP.items():
    _r, _g, _b = _color
    _CLASS_LUT[_r * 65536 + _g * 256 + _b] = _class_id

# AES-256 key from the project .env at the repository root, never hardcoded.
SECRET_KEY = dataset_key()
CONFIG_TAG = "calibrated_rgb_depth_semantic_secure_v1"

# Privatised calibration profile, the same for every town

CALIBRATION_PROFILE = {
    "vehicle_density_per_km": 2.5139,
    "pedestrian_density_per_km": 0.9153,
    "avg_vehicle_speed_kmh": 22.0135,
    "avg_pedestrian_speed_kmh": 0.2789,
    "intersection_count_per_km": 0.2796,
    "weather_distribution": {
        "clear": 0.6241,
        "cloudy": 0.135,
        "rain": 0.2409,
        "fog": 0.0
    },
    "time_of_day_distribution": {
        "morning": 0.1437,
        "afternoon": 0.1151,
        "evening": 0.5745,
        "night": 0.1666
    },
    # Total over the 7 queries; deliveries before the correction recorded 1.8 here.
    "calibration_epsilon_spent": 12.6,
    "calibration_epsilon_per_query": 1.8,
    "num_queries": 7,
    "generated_at": "2026-05-11T08:38:59.830695+00:00",
    "note": (
        "All values Laplace-noised. No raw sensor data included. "
        "Safe to share with simulator operator (IT22541284)."
    ),
}

# Town-specific settings

TOWN_PROFILES = {
    "town01": {
        "ego_vehicle": "vehicle.tesla.model3",
        "brake_threshold": 0.60,
        "steer_threshold": 0.15,
        "spectator_z": 40,
        "stuck_distance": 1.0,
        "stuck_check_ticks": 50,
        "stuck_strikes_limit": 4,
        "high_speed_threshold": None,
        "high_speed_label": ""
    },
    "town02": {
        "ego_vehicle": "vehicle.mini.cooper_s",
        "brake_threshold": 0.80,
        "steer_threshold": 0.15,
        "spectator_z": 40,
        "stuck_distance": 0.5,
        "stuck_check_ticks": 50,
        "stuck_strikes_limit": 4,
        "high_speed_threshold": None,
        "high_speed_label": ""
    },
    "town03": {
        "ego_vehicle": "vehicle.tesla.model3",
        "brake_threshold": 0.80,
        "steer_threshold": 0.15,
        "spectator_z": 45,
        "stuck_distance": 0.5,
        "stuck_check_ticks": 50,
        "stuck_strikes_limit": 5,
        "high_speed_threshold": None,
        "high_speed_label": ""
    },
    "town04": {
        "ego_vehicle": "vehicle.tesla.model3",
        "brake_threshold": 0.80,
        "steer_threshold": 0.12,
        "spectator_z": 50,
        "stuck_distance": 0.2,
        "stuck_check_ticks": 100,
        "stuck_strikes_limit": 2,
        "high_speed_threshold": 80,
        "high_speed_label": "High_Speed_Cruise"
    },
    "town05": {
        "ego_vehicle": "vehicle.tesla.model3",
        "brake_threshold": 0.80,
        "steer_threshold": 0.15,
        "spectator_z": 50,
        "stuck_distance": 0.5,
        "stuck_strikes_limit": 6,
        "stuck_check_ticks": 50,
        "high_speed_threshold": 90,
        "high_speed_label": "High_Speed_Flow"
    },
}

TOWN_SETTINGS = TOWN_PROFILES[TOWN_NAME]

# Thread-safe runtime state

sets_completed = 0
files_completed = 0
frames_dropped = 0
hash_manifest = {}
pending_records = {}
record_lock = threading.Lock()

# Helpers

def calculate_road_length_km(world, waypoint_step=2.0):
    waypoints = world.get_map().generate_waypoints(waypoint_step)
    return (len(waypoints) * waypoint_step) / 1000.0


def calculate_intersections(world, waypoint_step=2.0):
    waypoints = world.get_map().generate_waypoints(waypoint_step)
    junction_ids = set()

    for wp in waypoints:
        if wp.is_junction:
            try:
                junction_ids.add(wp.get_junction().id)
            except Exception:
                junction_ids.add((wp.road_id, wp.section_id))

    return len(junction_ids)


def build_camera_intrinsics(camera_bp, image_w, image_h):
    fov = float(camera_bp.get_attribute("fov").as_float())
    focal = image_w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K


def world_point_to_image(world_loc, K, world_to_camera):
    point = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    point_camera = np.dot(world_to_camera, point)

    # CARLA camera space (x-forward, y-right, z-up) -> standard image convention
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    point_img = np.dot(K, point_camera)
    if point_img[2] == 0:
        return None, False

    depth = point_img[2]
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return (point_img[0], point_img[1]), depth > 0.0


def actor_class_name(actor):
    type_id = actor.type_id

    if type_id.startswith("walker.pedestrian"):
        return "pedestrian"

    if type_id.startswith("vehicle."):
        if type_id in TWO_WHEELER_BLUEPRINTS:
            return "cyclist"
        return "car"

    return None


def compute_2d_bbox(actor, K, world_to_camera, image_w, image_h, camera_transform, max_distance):
    class_name = actor_class_name(actor)
    if class_name is None:
        return None

    actor_loc = actor.get_transform().location
    if actor_loc.distance(camera_transform.location) > max_distance:
        return None

    try:
        verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    except Exception:
        return None

    xs, ys = [], []
    any_in_front = False

    for v in verts:
        img_pt, in_front = world_point_to_image(v, K, world_to_camera)
        if img_pt is None:
            continue
        if in_front:
            any_in_front = True
        xs.append(img_pt[0])
        ys.append(img_pt[1])

    if not any_in_front or not xs or not ys:
        return None

    x_min, x_max = max(0.0, min(xs)), min(float(image_w), max(xs))
    y_min, y_max = max(0.0, min(ys)), min(float(image_h), max(ys))

    if x_max <= x_min or y_max <= y_min:
        return None

    pixel_area = (x_max - x_min) * (y_max - y_min)
    if pixel_area < BBOX_MIN_PIXEL_AREA:
        return None

    x_center = ((x_min + x_max) / 2.0) / image_w
    y_center = ((y_min + y_max) / 2.0) / image_h
    width = (x_max - x_min) / image_w
    height = (y_max - y_min) / image_h

    return {
        "class_id": CLASS_IDS[class_name],
        "class_name": class_name,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }


def write_bbox_label_file(path, bbox_actors, ego_vehicle, K, camera_actor, image_w, image_h):
    """Main-thread-only version: uses cached vehicle/walker actor lists."""
    world_to_camera = np.array(camera_actor.get_transform().get_inverse_matrix())
    camera_transform = camera_actor.get_transform()
    cam_loc = camera_transform.location
    max_dist_sq = BBOX_MAX_DISTANCE_M ** 2

    lines = []
    for actor in bbox_actors:
        if actor is None or (ego_vehicle is not None and actor.id == ego_vehicle.id):
            continue

        try:
            if not actor.is_alive:
                continue
            loc = actor.get_transform().location
        except Exception:
            # Actor destroyed mid-run; skip it rather than abort the frame.
            continue

        dx = loc.x - cam_loc.x
        dy = loc.y - cam_loc.y
        dz = loc.z - cam_loc.z
        if (dx * dx + dy * dy + dz * dz) > max_dist_sq:
            continue

        box = compute_2d_bbox(
            actor, K, world_to_camera, image_w, image_h, camera_transform, BBOX_MAX_DISTANCE_M
        )
        if box is None:
            continue

        lines.append(
            f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
            f"{box['width']:.6f} {box['height']:.6f}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(lines)


def remap_semantic_to_class_mask(semantic_png_path, output_path):
    """Convert a semantic PNG to a class mask: 0 car, 1 pedestrian, 2 cyclist, 3 background."""
    img = Image.open(semantic_png_path).convert("RGB")
    arr = np.asarray(img, dtype=np.int32)

    encoded = arr[:, :, 0] * 65536 + arr[:, :, 1] * 256 + arr[:, :, 2]
    mask = _CLASS_LUT[encoded]

    Image.fromarray(mask, mode="L").save(output_path, optimize=False, compress_level=1)


def build_exact_sequence(distribution, total_count):
    raw_counts = {key: value * total_count for key, value in distribution.items()}
    base_counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remaining = total_count - sum(base_counts.values())

    fractions = sorted(
        raw_counts.items(),
        key=lambda x: x[1] - math.floor(x[1]),
        reverse=True
    )

    for i in range(remaining):
        key = fractions[i % len(fractions)][0]
        base_counts[key] += 1

    sequence = []
    for key, count in base_counts.items():
        sequence.extend([key] * count)

    random.shuffle(sequence)
    return sequence


def weather_params_from_categories(weather_category, time_category):
    sun_map = {
        "morning": 15,
        "afternoon": 70,
        "evening": 5,
        "night": -30
    }

    sun = sun_map.get(time_category, 70)

    if weather_category == "clear":
        return {
            "cloudiness": 5, "precipitation": 0, "fog_density": 0, "wetness": 0,
            "sun_altitude_angle": sun,
        }

    if weather_category == "cloudy":
        return {
            "cloudiness": 80, "precipitation": 0, "fog_density": 5, "wetness": 10,
            "sun_altitude_angle": sun,
        }

    if weather_category == "rain":
        return {
            "cloudiness": 90, "precipitation": 80, "fog_density": 10, "wetness": 100,
            "sun_altitude_angle": sun,
        }

    if weather_category == "fog":
        return {
            "cloudiness": 60, "precipitation": 0, "fog_density": 60, "wetness": 20,
            "sun_altitude_angle": sun,
        }

    return {
        "cloudiness": 5, "precipitation": 0, "fog_density": 0, "wetness": 0,
        "sun_altitude_angle": sun,
    }


def build_scenario_schedule(num_sets):
    weather_sequence = build_exact_sequence(CALIBRATION_PROFILE["weather_distribution"], num_sets)
    time_sequence = build_exact_sequence(CALIBRATION_PROFILE["time_of_day_distribution"], num_sets)

    schedule = []
    for i in range(num_sets):
        weather_category = weather_sequence[i]
        time_category = time_sequence[i]
        schedule.append({
            "weather_category": weather_category,
            "time_of_day": time_category,
            "weather_label": f"{weather_category}_{time_category}",
            "params": weather_params_from_categories(weather_category, time_category)
        })

    return schedule


def speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * ((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)


def get_behavior_event(vehicle, world):
    try:
        waypoint = world.get_map().get_waypoint(vehicle.get_location())
        control = vehicle.get_control()
        speed = speed_kmh(vehicle)

        if waypoint.is_junction:
            return "Intersection_Nav"

        if abs(control.steer) > TOWN_SETTINGS["steer_threshold"]:
            return "Vehicle_Turning"

        if control.brake > TOWN_SETTINGS["brake_threshold"]:
            return "Heavy_Braking"

        hs_threshold = TOWN_SETTINGS.get("high_speed_threshold")
        if hs_threshold is not None and speed > hs_threshold:
            return TOWN_SETTINGS.get("high_speed_label", "High_Speed_Flow")

        return "Standard_Flow"

    except Exception:
        return "Unknown"


def apply_calibrated_vehicle_speed(traffic_manager):
    target_speed = CALIBRATION_PROFILE["avg_vehicle_speed_kmh"]

    assumed_default_speed = 30.0
    percentage_slowdown = ((assumed_default_speed - target_speed) / assumed_default_speed) * 100.0
    percentage_slowdown = max(-50.0, min(90.0, percentage_slowdown))

    with contextlib.suppress(Exception):
        traffic_manager.global_percentage_speed_difference(percentage_slowdown)

    return percentage_slowdown


def spawn_walkers(world, bp_lib, count):
    walkers = []
    controllers = []

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    target_ped_speed_mps = CALIBRATION_PROFILE["avg_pedestrian_speed_kmh"] / 3.6
    target_ped_speed_mps = max(0.2, target_ped_speed_mps)

    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if not location:
            continue

        try:
            walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(location))
            if walker:
                walkers.append(walker)
        except Exception:
            pass

    world.tick()

    for walker in walkers:
        try:
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(target_ped_speed_mps)
                controllers.append(controller)
        except Exception:
            pass

    print(f"  Walkers spawned  : {len(walkers)} / requested {count}")
    return walkers, controllers


def spawn_traffic(world, bp_lib, spawn_points, count, traffic_manager):
    vehicles = []
    vehicle_bps = bp_lib.filter("vehicle.*")

    for _ in range(count):
        try:
            vehicle = world.try_spawn_actor(random.choice(vehicle_bps), random.choice(spawn_points))
            if vehicle:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles.append(vehicle)
        except Exception:
            pass

    print(f"  Vehicles spawned : {len(vehicles)} / requested {count}")
    return vehicles


def spawn_ego_vehicle(world, bp_lib, spawn_points, traffic_manager):
    preferred = TOWN_SETTINGS["ego_vehicle"]
    possible_vehicles = [preferred, "vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz_2020"]

    for vehicle_id in possible_vehicles:
        try:
            ego_bp = bp_lib.find(vehicle_id)
        except Exception:
            continue

        for _ in range(30):
            try:
                ego = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
                if ego:
                    ego.set_autopilot(True, traffic_manager.get_port())
                    print(f"  Ego vehicle spawned: {vehicle_id}")
                    return ego
            except Exception:
                pass

    raise RuntimeError("Could not spawn ego vehicle. Restart CARLA or reduce traffic.")


def safe_stop_sensor(sensor):
    try:
        if sensor and sensor.is_alive:
            sensor.stop()
    except Exception:
        pass


def safe_stop_controller(controller):
    try:
        if controller and controller.is_alive:
            controller.stop()
    except Exception:
        pass


def safe_destroy_actor(world, actor):
    try:
        if actor is None:
            return
        live_actor = world.get_actor(actor.id)
        if live_actor is not None and live_actor.is_alive:
            live_actor.destroy()
    except Exception:
        pass


def save_carla_image(image, out_path, sensor_name):
    """Save CARLA image to PNG in the callback thread only."""
    if sensor_name == "semantic" and SAVE_SEMANTIC_AS_CITYSCAPES:
        image.save_to_disk(out_path, carla.ColorConverter.CityScapesPalette)
    else:
        image.save_to_disk(out_path)


# Main capture pipeline

def main():
    global sets_completed, files_completed, hash_manifest, pending_records

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    rbac = RBACGateway()
    if not rbac.verify_access(USER_ROLE, "create"):
        return

    crypto = SecurityEngine(key=SECRET_KEY)

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    world = None
    settings = None
    traffic_manager = None

    ego_v = None
    rgb_cam = None
    dep_cam = None
    sem_cam = None

    traffic_actors = []
    walkers = []
    controllers = []

    metadata_path = os.path.join(OUTPUT_PATH, f"{TOWN_NAME}_metadata.csv")
    manifest_path = os.path.join(OUTPUT_PATH, f"{TOWN_NAME}_manifest.json")
    config_path = os.path.join(OUTPUT_PATH, f"{TOWN_NAME}_config_metadata.json")

    try:
        print(f"\nLoading {CARLA_TOWN}...")
        world = client.load_world(CARLA_TOWN)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        speed_slowdown = apply_calibrated_vehicle_speed(traffic_manager)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        road_length_km = calculate_road_length_km(world)
        intersection_count = calculate_intersections(world)

        requested_vehicles = max(1, round(CALIBRATION_PROFILE["vehicle_density_per_km"] * road_length_km))
        requested_walkers = max(1, round(CALIBRATION_PROFILE["pedestrian_density_per_km"] * road_length_km))

        scenario_schedule = build_scenario_schedule(NUM_SETS)

        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "RGB_File",
                "Depth_File",
                "Semantic_File",
                "ClassMask_File",
                "BBox_File",
                "Num_Objects",
                "Weather_Category",
                "Time_Of_Day",
                "Weather_Label",
                "Event_Type",
                "Town",
                "Security"
            ])

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        ego_v = spawn_ego_vehicle(world, bp_lib, spawn_points, traffic_manager)
        traffic_actors = spawn_traffic(world, bp_lib, spawn_points, requested_vehicles, traffic_manager)
        walkers, controllers = spawn_walkers(world, bp_lib, requested_walkers)

        bbox_actors = traffic_actors + walkers

        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        dep_bp = bp_lib.find("sensor.camera.depth")
        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")

        rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego_v)
        dep_cam = world.spawn_actor(dep_bp, cam_tf, attach_to=ego_v)
        sem_cam = world.spawn_actor(sem_bp, cam_tf, attach_to=ego_v)

        image_width = int(rgb_bp.get_attribute("image_size_x").as_int())
        image_height = int(rgb_bp.get_attribute("image_size_y").as_int())
        camera_K = build_camera_intrinsics(rgb_bp, image_width, image_height)

        bbox_dir = os.path.join(OUTPUT_PATH, "labels")
        if SAVE_BOUNDING_BOXES:
            os.makedirs(bbox_dir, exist_ok=True)

        class_mask_dir = os.path.join(OUTPUT_PATH, "class_masks")
        if SAVE_CLASS_MASKS:
            os.makedirs(class_mask_dir, exist_ok=True)

        def create_pending_record(frame_id):
            if frame_id not in pending_records:
                if sets_completed >= NUM_SETS:
                    return None
                scenario_index = min(sets_completed, NUM_SETS - 1)
                pending_records[frame_id] = {
                    "files": {},
                    "paths": {},
                    "scenario": scenario_schedule[scenario_index],
                    "event": "",
                    "class_mask_file": "",
                    "bbox_file": "",
                    "num_objects": 0,
                    "processed": False,
                }
            return pending_records[frame_id]

        def sensor_callback(image, sensor_name):
            if image.frame % CAPTURE_INTERVAL != 0:
                return

            img_name = f"{sensor_name}_f{image.frame:06d}.png"
            full_path = os.path.join(OUTPUT_PATH, img_name)

            with record_lock:
                if sets_completed >= NUM_SETS:
                    return
                rec = create_pending_record(image.frame)
                if rec is None:
                    return

            try:
                save_carla_image(image, full_path, sensor_name)
            except Exception as e:
                print(f"\nERROR: Failed to save {sensor_name} frame {image.frame}: {e}")
                return

            with record_lock:
                rec = pending_records.get(image.frame)
                if rec is None:
                    return
                rec["files"][sensor_name] = img_name
                rec["paths"][sensor_name] = full_path

        def process_completed_frames():
            nonlocal bbox_actors
            global sets_completed, files_completed, hash_manifest, pending_records

            while True:
                with record_lock:
                    ready_frame = None
                    for frame_id in sorted(pending_records.keys()):
                        rec = pending_records[frame_id]
                        if rec.get("processed"):
                            continue
                        if all(sensor in rec["files"] for sensor in REQUIRED_SENSORS):
                            ready_frame = frame_id
                            rec["processed"] = True
                            break

                    if ready_frame is None:
                        return

                    rec = pending_records[ready_frame]

                frame_id = ready_frame
                scenario = rec["scenario"]
                event_label = get_behavior_event(ego_v, world) if ego_v is not None else "Unknown"

                rgb_path = rec["paths"]["rgb"]
                depth_path = rec["paths"]["depth"]
                semantic_path = rec["paths"]["semantic"]

                rgb_name = rec["files"]["rgb"]
                depth_name = rec["files"]["depth"]
                semantic_name = rec["files"]["semantic"]

                rgb_hash = depth_hash = semantic_hash = ""
                rgb_encrypted = depth_encrypted = semantic_encrypted = ""
                class_mask_name = ""
                class_mask_encrypted_name = ""
                bbox_file_name = ""
                num_objects = 0

                try:
                    # Hash + encrypt sensor files.
                    rgb_hash = crypto.generate_sha256(rgb_path)
                    if rgb_hash:
                        crypto.encrypt_data(rgb_path)
                        rgb_encrypted = rgb_name + ".enc"

                    depth_hash = crypto.generate_sha256(depth_path)
                    if depth_hash:
                        crypto.encrypt_data(depth_path)
                        depth_encrypted = depth_name + ".enc"

                    semantic_hash = crypto.generate_sha256(semantic_path)

                    # Build the class mask before the semantic PNG is encrypted and removed.
                    if SAVE_CLASS_MASKS:
                        class_mask_name = f"class_mask_f{frame_id:06d}.png"
                        class_mask_path = os.path.join(class_mask_dir, class_mask_name)
                        remap_semantic_to_class_mask(semantic_path, class_mask_path)
                        class_mask_hash = crypto.generate_sha256(class_mask_path)
                        if class_mask_hash:
                            crypto.encrypt_data(class_mask_path)
                            class_mask_encrypted_name = class_mask_name + ".enc"
                            with record_lock:
                                hash_manifest[class_mask_name] = class_mask_hash

                    if semantic_hash:
                        crypto.encrypt_data(semantic_path)
                        semantic_encrypted = semantic_name + ".enc"

                    # Bounding boxes are written in the main thread only.
                    if SAVE_BOUNDING_BOXES:
                        bbox_file_name = f"bbox_f{frame_id:06d}.txt"
                        bbox_path = os.path.join(bbox_dir, bbox_file_name)
                        num_objects = write_bbox_label_file(
                            bbox_path,
                            bbox_actors,
                            ego_v,
                            camera_K,
                            rgb_cam,
                            image_width,
                            image_height
                        )

                    # Update manifest for raw files.
                    with record_lock:
                        if rgb_hash:
                            hash_manifest[rgb_name] = rgb_hash
                        if depth_hash:
                            hash_manifest[depth_name] = depth_hash
                        if semantic_hash:
                            hash_manifest[semantic_name] = semantic_hash

                    # Write metadata row.
                    with open(metadata_path, "a", newline="", encoding="utf-8") as csv_file:
                        csv.writer(csv_file).writerow([
                            rgb_encrypted,
                            depth_encrypted,
                            semantic_encrypted,
                            class_mask_encrypted_name,
                            bbox_file_name,
                            num_objects,
                            scenario["weather_category"],
                            scenario["time_of_day"],
                            scenario["weather_label"],
                            event_label,
                            CARLA_TOWN,
                            "SHA256+AES256"
                        ])

                    with record_lock:
                        files_completed += 3
                        sets_completed += 1
                        del pending_records[frame_id]

                    print(
                        f"  Frame {frame_id:06d} processed | "
                        f"sets {sets_completed}/{NUM_SETS} | "
                        f"files {files_completed}/{NUM_SETS * SENSORS_PER_SET}",
                        end="\r"
                    )

                except Exception as e:
                    # Not idempotent: files may already be encrypted, so drop this frame and continue.
                    global frames_dropped
                    print(f"\nERROR: Dropping frame {frame_id} after processing failure: {e}")
                    print(traceback.format_exc())
                    with record_lock:
                        frames_dropped += 1
                        if frame_id in pending_records:
                            del pending_records[frame_id]
                    continue

        rgb_cam.listen(lambda data: sensor_callback(data, "rgb"))
        dep_cam.listen(lambda data: sensor_callback(data, "depth"))
        sem_cam.listen(lambda data: sensor_callback(data, "semantic"))

        config_metadata = {
            "config_tag": CONFIG_TAG,
            "town": CARLA_TOWN,
            "output_name": TOWN_NAME,
            "num_sets": NUM_SETS,
            "capture_interval_frames": CAPTURE_INTERVAL,
            "fixed_delta_seconds": 0.05,
            "sensors": ["rgb", "depth", "semantic_segmentation"],
            "sensor_shapes": {
                "rgb": "PNG RGB camera frame",
                "depth": "PNG depth camera frame",
                "semantic": "PNG semantic segmentation frame using CityScapesPalette"
            },
            "ground_truth_bounding_boxes": {
                "enabled": SAVE_BOUNDING_BOXES,
                "format": "YOLO txt: class_id x_center_norm y_center_norm width_norm height_norm",
                "folder": "labels/",
                "filename_pattern": "bbox_f{frame:06d}.txt",
                "class_ids": CLASS_IDS,
                "max_distance_m": BBOX_MAX_DISTANCE_M,
                "min_pixel_area": BBOX_MIN_PIXEL_AREA,
                "visibility_method": (
                    "camera frustum + distance filter only (not full depth-buffer occlusion)"
                ),
                "performance_note": "bbox computation runs in the main thread only; actor lists are cached"
            },
            "semantic_class_masks": {
                "enabled": SAVE_CLASS_MASKS,
                "format": "single-channel PNG, pixel value = class ID",
                "folder": "class_masks/",
                "filename_pattern": "class_mask_f{frame:06d}.png",
                "class_ids": CLASS_NAMES_WITH_BACKGROUND,
                "source_palette": "CARLA CityScapesPalette (0.9.14+ tag/color table)",
                "note": (
                    "Trucks and buses are grouped into the car class; "
                    "motorcycles are grouped into cyclist."
                ),
            },
            "calibration_profile": CALIBRATION_PROFILE,
            "calculated_map_stats": {
                "road_length_km": round(road_length_km, 4),
                "intersection_count": intersection_count,
                "intersection_count_per_km_observed": (
                    round(intersection_count / road_length_km, 4) if road_length_km > 0 else 0
                ),
            },
            "spawn_plan_from_calibration": {
                "requested_vehicles": requested_vehicles,
                "requested_walkers": requested_walkers,
                "vehicle_speed_slowdown_percent": round(speed_slowdown, 4)
            },
            # Per query, as in every delivery so far; epsilon_total is the composed total.
            "epsilon_reference": CALIBRATION_PROFILE["calibration_epsilon_per_query"],
            "epsilon_total": CALIBRATION_PROFILE["calibration_epsilon_spent"],
            "security": {
                "hashing": "SHA-256",
                "encryption": "AES-256",
                "rbac_role": USER_ROLE,
                "raw_png_removed_after_encryption": True
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_metadata, f, indent=4)

        last_loc = ego_v.get_location()
        stuck_timer = 0
        stuck_strikes = 0

        print(f"\n{CARLA_TOWN} Calibrated Secure Capture Started")
        print(f"    Output folder        : {OUTPUT_PATH}")
        print(f"    Target sets          : {NUM_SETS}")
        print(f"    Target files         : {NUM_SETS * SENSORS_PER_SET}")
        print("    Sensors              : RGB + Depth + Semantic")
        print(f"    Road length km       : {road_length_km:.4f}")
        print(f"    Requested vehicles   : {requested_vehicles}")
        print(f"    Requested walkers    : {requested_walkers}")
        print(f"    Spawned vehicles     : {len(traffic_actors)}")
        print(f"    Spawned walkers      : {len(walkers)}")
        print(f"    Speed slowdown       : {speed_slowdown:.2f}%")
        print(f"    Weather distribution : {CALIBRATION_PROFILE['weather_distribution']}")
        print(f"    Time distribution    : {CALIBRATION_PROFILE['time_of_day_distribution']}\n")

        while True:
            with record_lock:
                done = sets_completed >= NUM_SETS
                progress_sets = sets_completed
                progress_files = files_completed

            if done:
                break

            world.tick()

            scenario = scenario_schedule[min(progress_sets, NUM_SETS - 1)]
            params = scenario["params"]

            world.set_weather(carla.WeatherParameters(
                cloudiness=params["cloudiness"],
                precipitation=params["precipitation"],
                sun_altitude_angle=params["sun_altitude_angle"],
                fog_density=params["fog_density"],
                wetness=params["wetness"]
            ))

            if ego_v and ego_v.is_alive:
                loc = ego_v.get_transform().location
                world.get_spectator().set_transform(
                    carla.Transform(
                        loc + carla.Location(z=TOWN_SETTINGS["spectator_z"]),
                        carla.Rotation(pitch=-90)
                    )
                )

            stuck_timer += 1
            if stuck_timer >= TOWN_SETTINGS["stuck_check_ticks"]:
                stuck_timer = 0
                current_loc = ego_v.get_location()

                if current_loc.distance(last_loc) < TOWN_SETTINGS["stuck_distance"]:
                    stuck_strikes += 1
                    limit = TOWN_SETTINGS["stuck_strikes_limit"]
                    print(f"\n  WARNING: Stuck warning {stuck_strikes}/{limit}")

                    if stuck_strikes >= TOWN_SETTINGS["stuck_strikes_limit"]:
                        ego_v.set_transform(random.choice(spawn_points))
                        ego_v.set_autopilot(True, traffic_manager.get_port())
                        stuck_strikes = 0
                        print("  Ego vehicle teleported and autopilot restored")
                else:
                    stuck_strikes = 0

                last_loc = current_loc

            event = get_behavior_event(ego_v, world)

            print(
                f"  {progress_sets:>5}/{NUM_SETS} sets | "
                f"{progress_files:>5}/{NUM_SETS * SENSORS_PER_SET} files | "
                f"{scenario['weather_label']:<15} | {event}",
                end="\r"
            )

            process_completed_frames()

        # Final pass in case the last few sensor callbacks arrived just before exit.
        for _ in range(10):
            process_completed_frames()
            with record_lock:
                if not pending_records:
                    break
            time.sleep(0.1)

        time.sleep(1)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(hash_manifest, f, indent=4)

        print(f"\n\n{CARLA_TOWN} Complete!")
        print(f"    Sets secured   : {sets_completed}")
        print(f"    Files secured  : {len(hash_manifest)}")
        print(f"    Frames dropped : {frames_dropped}  (processing failures -- see log above for causes)")
        print(f"    Metadata CSV   : {metadata_path}")
        print(f"    Manifest JSON  : {manifest_path}")
        print(f"    Config JSON    : {config_path}")

    finally:
        print("\nCleaning up actors safely...")

        for sensor in [rgb_cam, dep_cam, sem_cam]:
            safe_stop_sensor(sensor)

        for controller in controllers:
            safe_stop_controller(controller)

        if world:
            with contextlib.suppress(Exception):
                world.tick()

            for actor in [rgb_cam, dep_cam, sem_cam, ego_v]:
                safe_destroy_actor(world, actor)

            for actor in controllers + walkers + traffic_actors:
                safe_destroy_actor(world, actor)

        try:
            if settings and world:
                settings.synchronous_mode = False
                world.apply_settings(settings)
        except Exception:
            pass

        try:
            if traffic_manager:
                traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

        print("Cleanup complete")


if __name__ == "__main__":
    main()
