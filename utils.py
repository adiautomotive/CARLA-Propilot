import pygame
import carla
import numpy as np
import evdev
from evdev import ecodes, ff
# Carla PythonAPI path
import sys
sys.path.append('/home/jesudara/carla_dev/carla/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
import math

pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()



import re

def camel_case(s):
    return re.sub(r"(\b[a-z])", lambda x: x.group(1).upper(), s)

def to_human_readable(vehicles):
    if isinstance(vehicles, str):
        vehicles = [vehicles]  # Wrap single string into a list

    human_readable = []
    for v in vehicles:
        parts = v.split(".")[1:]  # Skip "vehicle"
        if len(parts) < 2:
            human_readable.append("Unknown Vehicle")
            continue
        make = parts[0].capitalize()
        model = parts[1]
        model = re.sub(r'(\d+)', r' \1', model)  # Add space before digits
        model = model.replace('_', ' ')
        model = re.sub(r'([a-z])([A-Z])', r'\1 \2', model)  # Handle camelCase if needed
        model = model.title()  # Capitalize each word
        human_readable.append(f"{make} {model}")
    return human_readable if len(human_readable) > 1 else human_readable[0]

def to_vehicle_code(names):
    if isinstance(names, str):
        names = [names]  # Wrap single string into a list

    vehicle_code = []
    for name in names:
        parts = name.strip().split(" ", 1)
        if len(parts) < 2:
            vehicle_code.append("vehicle.unknown.unknown")
            continue
        make = parts[0].lower()
        model = parts[1]
        model = model.replace(" ", "")  # Remove spaces
        model = model.lower()
        vehicle_code.append(f"vehicle.{make}.{model}")
    
    return vehicle_code if len(vehicle_code) > 1 else vehicle_code[0]
def get_weather_parameter(weather_name, weather_options):
    for name, param in weather_options:
        if name == weather_name:
            return param
    raise ValueError(f"Weather option '{weather_name}' not found.")
# === Control Logic ===
def get_joystick_control(joystick):
    pygame.event.pump()
    control = carla.VehicleControl()
    control.throttle = max(0.0, -2*joystick.get_axis(2))  # Right trigger
    control.brake = max(0.0, -joystick.get_axis(5))    # Left trigger
    control.steer = joystick.get_axis(0)     # Left stick X
    control.reverse = joystick.get_button(4)
    return control

def get_keyboard_control(keys):
    control = carla.VehicleControl()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        control.throttle = 1.0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        control.brake = 1.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        control.steer = -1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        control.steer = 1.0
    return control


# === Camera Rending ===

def render_camera_image(image, display):
    img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_bgra = img_bgra.reshape((image.height, image.width, 4))
    img_rgb = img_bgra[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB

    # Slight desaturation for natural color, tone down blue channel
    img_rgb = img_rgb.astype(np.float32)
    img_rgb[:, :, 0] *= 1.05   # Red
    img_rgb[:, :, 1] *= 1.05   # Green
    img_rgb[:, :, 2] *= 0.85   # Blue
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
    # surface = pygame.transform.scale(surface, (screen_width, screen_height))
    display.blit(surface, (0, 0))
    

def switch_camera(world, vehicle, camera_bp, camera_angles, current_camera_idx,screen):
    current_camera_idx = (current_camera_idx + 1) % len(camera_angles)
    camera = world.get_actors().filter('sensor.camera.rgb')[0]
    camera.stop()
    camera.destroy()
    new_camera = world.spawn_actor(camera_bp, camera_angles[current_camera_idx], attach_to=vehicle)
    new_camera.listen(lambda image: render_camera_image(image, screen))
    return new_camera, current_camera_idx

def setup_camera(self, camera_angles):
    self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
    self.camera_bp.set_attribute('image_size_x', str(SCREEN_WIDTH))
    self.camera_bp.set_attribute('image_size_y', str(SCREEN_HEIGHT))
    transform = camera_angles[self.camera_idx]
    self.camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=self.vehicle)
    self.camera.listen(lambda image: render_camera_image(image, self.screen))


# === Weather Rending ===
def cycle_weather(world,WEATHER_PRESETS, current_weather_idx):
    new_weather = WEATHER_PRESETS[current_weather_idx]
    world.set_weather(new_weather)
    print(f"[INFO] Weather changed to: {new_weather}")


# === Navigation/Path Planning ===
def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

def get_angle(vehicle, waypoint):
    tf = vehicle.get_transform()
    dx = waypoint.transform.location.x - tf.location.x
    dy = waypoint.transform.location.y - tf.location.y
    direction = (dx / math.hypot(dx, dy), dy / math.hypot(dx, dy))
    forward = tf.get_forward_vector()
    return angle_between(direction, (forward.x, forward.y))

def plan_route(start_loc, world_map, spawn_points):
    planner = GlobalRoutePlanner(world_map, 1)
    longest_route = []
    max_len = 0
    for sp in spawn_points[1:]:
        route = planner.trace_route(start_loc, sp.location)
        if len(route) > max_len:
            longest_route = route
            max_len = len(route)
    return longest_route

def maintain_speed(current_speed, PREFERRED_SPEED, SPEED_THRESHOLD):
    if current_speed >= PREFERRED_SPEED:
        return 0.0
    elif current_speed < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9
    else:
        return 0.4
    

# === Force Feedback Effect ===    
def setup_force_feedback(device_path="/dev/input/by-id/usb-Fanatec_FANATEC_CSL_Elite_Wheel_Base-event-joystick"):
    try:
        ff_device = evdev.InputDevice(device_path)
        print(f"[INFO] Force Feedback device: {ff_device.name}")

        # Maximum values for stiffness and damping
        strong_condition = ff.Condition(
            right_saturation=0x7FFF,  # Max force
            left_saturation=0x7FFF,
            right_coeff=0x7FFF,       # Stiffness
            left_coeff=0x7FFF,
            deadband=0,
            center=0
        )

        ff_effect = ff.Effect(
            ecodes.FF_SPRING,
            -1,  # id
            0,   # direction
            ff.Trigger(0, 0),
            ff.Replay(0xFFFF, 0),  # Effect should last indefinitely
            ff.EffectType(ff_condition_effect=(strong_condition, strong_condition))
        )

        effect_id = ff_device.upload_effect(ff_effect)
        ff_device.write(ecodes.EV_FF, effect_id, 1)
        print("[INFO] Spring force feedback uploaded and started.")

        return ff_device, effect_id

    except Exception as e:
        print(f"[ERROR] Force feedback setup failed: {e}")
        return None, None

# === UI Functions ===
def draw_tor_popup(SCREEN_WIDTH, SCREEN_HEIGHT,screen, font):
    
    text = font.render("TAKEOVER REQUEST!", True, (255, 0, 0))
    screen.blit(text, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))

def draw_text(surface, text, x, y, font, color=(255, 255, 255)):
    label = font.render(text, True, color)
    surface.blit(label, (x, y))

# WIDTH, HEIGHT = 1920, 1080
MINIMAP_SIZE = (160, 90)
# MINIMAP_POS = (1750, 980)
# MAP_EXTENT = (300, 300)
# ==== FUNCTION: Spawn Minimap Camera ====
def spawn_minimap_camera(world, map, z_height=150):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(MINIMAP_SIZE[0]))
    camera_bp.set_attribute('image_size_y', str(MINIMAP_SIZE[1]))
    camera_bp.set_attribute('fov', '70')

    map_center = map.get_spawn_points()[0].location
    cam_location = carla.Location(x=map_center.x, y=map_center.y, z=z_height)
    cam_rotation = carla.Rotation(pitch=-90)  # top-down

    transform = carla.Transform(cam_location, cam_rotation)
    camera = world.spawn_actor(camera_bp, transform)
    return camera, cam_location

# ==== FUNCTION: Process Minimap Image ====
def process_minimap_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return array[:, :, :3][:, :, ::-1]  # RGB

# ==== FUNCTION: Draw Vehicle Marker on Minimap ====
def draw_vehicle_marker_on_minimap(surface, ego_location, cam_center, map_extent, surface_size):
    sw, sh = surface_size
    mw, mh = map_extent
    dx = ego_location.x - cam_center.x
    dy = ego_location.y - cam_center.y

    px = int((dx / mw + 0.5) * sw)
    py = int((-dy / mh + 0.5) * sh)
    px = max(0, min(sw - 1, px))
    py = max(0, min(sh - 1, py))

    pygame.draw.circle(surface, (255, 0, 0), (px, py), 4)

class WaypointNavigator:
    def __init__(self, world, vehicle, resolution=1.0, max_steer_degrees=40, speed_threshold=2.0, preferred_speed=70.0):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.grp = GlobalRoutePlanner(self.map, resolution)
        self.route = []
        self.curr_wp_index = 0
        self.max_steer = max_steer_degrees
        self.speed_threshold = speed_threshold
        self.preferred_speed = preferred_speed

    def plan_to(self, destination):
        """Plan route from current location to destination location."""
        start_loc = self.vehicle.get_transform().location
        self.route = self.grp.trace_route(start_loc, destination)
        self.curr_wp_index = 0

        # Optional: visualize route in simulation
        for wp, _ in self.route:
            self.world.debug.draw_string(
                wp.transform.location, '^', draw_shadow=False,
                color=carla.Color(0, 255, 0), life_time=30.0, persistent_lines=True
            )

    def maintain_speed(self, speed):
        """Simple proportional speed control."""
        if speed >= self.preferred_speed:
            return 0.0
        elif speed < self.preferred_speed - self.speed_threshold:
            return 0.9
        else:
            return 0.4

    def get_angle_to_next_waypoint(self):
        """Compute angle between vehicle and current target waypoint."""
        if not self.route or self.curr_wp_index >= len(self.route):
            return 0.0

        tf = self.vehicle.get_transform()
        loc = tf.location
        wp_loc = self.route[self.curr_wp_index][0].transform.location

        dx = wp_loc.x - loc.x
        dy = wp_loc.y - loc.y
        direction = (dx / math.hypot(dx, dy), dy / math.hypot(dx, dy))
        forward = tf.get_forward_vector()

        return math.degrees(math.atan2(direction[1], direction[0]) - math.atan2(forward.y, forward.x))

    def update_waypoint_index(self, distance_threshold=5.0):
        """Advance to the next waypoint if close enough to the current."""
        while self.curr_wp_index < len(self.route):
            wp_loc = self.route[self.curr_wp_index][0].transform.location
            distance = self.vehicle.get_transform().location.distance(wp_loc)
            if distance >= distance_threshold:
                break
            self.curr_wp_index += 1

    @property
    def reached_destination(self):
        # Destination is reached if waypoint index is past last waypoint
        return self.curr_wp_index >= len(self.route)

    def run_step(self):
        """Compute control command (throttle, steer) to follow the route."""
        if not self.route or self.curr_wp_index >= len(self.route):
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        self.update_waypoint_index()

        angle = self.get_angle_to_next_waypoint()
        if angle < -300: angle += 360
        elif angle > 300: angle -= 360

        steer = max(min(angle, self.max_steer), -self.max_steer) / 75.0

        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        throttle = self.maintain_speed(speed)

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)





import carla
import math
import numpy as np

class TrafficLightNavigator:
    def __init__(self, world, vehicle, max_steer_degrees=50, preferred_speed=40, speed_threshold=1):
        self.world = world
        self.vehicle = vehicle
        self.max_steer_degrees = max_steer_degrees
        self.preferred_speed = preferred_speed
        self.speed_threshold = speed_threshold

    def run_step(self):
        control = carla.VehicleControl()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        traffic_light = self.vehicle.get_traffic_light()

        if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
            distance = transform.location.distance(traffic_light.get_transform().location)
            if distance < 15.0:  # Stop if close to red light
                control.throttle = 0.0
                control.brake = 1.0
                return control

        # Maintain preferred speed
        if speed >= self.preferred_speed:
            control.throttle = 0.0
        elif speed < self.preferred_speed - self.speed_threshold:
            control.throttle = 0.9
        else:
            control.throttle = 0.4

        # Basic forward driving with minimal steering (can be enhanced later)
        control.steer = 0.0
        return control
