# ==========================================
# Autonomous Waypoint Navigation in CARLA Simulator
# ==========================================

# 1. Required Imports
import carla
import time
import cv2
import numpy as np
import math
import sys

# Add CARLA PythonAPI path (Update this path if needed)
sys.path.append('/home/jesudara/carla_dev/carla/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

# 2. Simulation Connection & Parameters
client = carla.Client('141.215.211.243', 2000)
world = client.get_world()

# Vehicle and control parameters
PREFERRED_SPEED = 60         # Target speed in km/h
SPEED_THRESHOLD = 1          # Acceptable deviation before adjusting throttle
MAX_STEER_DEGREES = 40       # Max steering in degrees
CAMERA_POS_Z = 3             # Camera height
CAMERA_POS_X = -5            # Camera X offset (rearward)

# 3. Display Text Config (for on-screen telemetry)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 255, 255)
thickness = 1
org = (30, 30)
org2 = (30, 50)
org3 = (30, 110)

# 4. Utility Functions
def maintain_speed(current_speed):
    """Determine throttle based on current speed."""
    if current_speed >= PREFERRED_SPEED:
        return 0.0
    elif current_speed < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9
    else:
        return 0.4

def angle_between(v1, v2):
    """Returns angle between two 2D vectors in degrees."""
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

def get_angle(vehicle, waypoint):
    """Compute angle between vehicle heading and waypoint direction."""
    transform = vehicle.get_transform()
    car_loc = transform.location
    wp_loc = waypoint.transform.location

    # Vector to waypoint
    dx = wp_loc.x - car_loc.x
    dy = wp_loc.y - car_loc.y
    magnitude = math.sqrt(dx**2 + dy**2)
    direction_vec = (dx / magnitude, dy / magnitude)

    # Vehicle forward vector
    forward_vec = transform.get_forward_vector()
    car_vec = (forward_vec.x, forward_vec.y)

    return angle_between(direction_vec, car_vec)

# 5. Vehicle Spawning
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = world.get_blueprint_library().filter('*mini*')[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

# 6. Route Planning
point_a = spawn_points[0].location
grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=1)

# Find the longest route among spawn points
route = []
max_len = 0
for point in spawn_points[1:]:
    r = grp.trace_route(point_a, point.location)
    if len(r) > max_len:
        route = r
        max_len = len(r)

# Draw route in world (not visible in camera)
for wp, _ in route:
    world.debug.draw_string(
        wp.transform.location, '^', draw_shadow=False,
        color=carla.Color(r=0, g=0, b=255), life_time=30.0, persistent_lines=True
    )

# 7. Camera Setup
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '360')
camera_transform = carla.Transform(carla.Location(x=CAMERA_POS_X, z=CAMERA_POS_Z))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Live camera feed handling
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()
camera_data = {'image': np.zeros((image_h, image_w, 4))}
camera.listen(lambda image: camera_callback(image, camera_data))

cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)

# 8. Main Loop
curr_wp = 5  # Initial waypoint index to follow
quit_flag = False

while curr_wp < len(route) - 1:
    world.tick()

    # Quit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        quit_flag = True
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        break

    # Advance waypoint if close to current
    while curr_wp < len(route) and vehicle.get_transform().location.distance(route[curr_wp][0].transform.location) < 5:
        curr_wp += 1

    # Get vehicle speed (m/s to km/h)
    velocity = vehicle.get_velocity()
    speed = round(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2), 0)

    # Compute steering angle
    predicted_angle = get_angle(vehicle, route[curr_wp][0])

    # Normalize wrap-around angles
    if predicted_angle < -300:
        predicted_angle += 360
    elif predicted_angle > 300:
        predicted_angle -= 360

    # Clamp steering angle to max bounds
    steer_angle = np.clip(predicted_angle, -MAX_STEER_DEGREES, MAX_STEER_DEGREES)
    steer_input = steer_angle / 75  # Normalize to range [-1, 1]

    throttle = maintain_speed(speed)

    # Apply vehicle control
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_input))

    # Display telemetry
    image = camera_data['image']
    image = cv2.putText(image, f'Steering angle: {round(predicted_angle, 3)}', org, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, f'Speed: {int(speed)} km/h', org2, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, f'Next wp index: {curr_wp}', org3, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('RGB Camera', image)

# 9. Cleanup
cv2.destroyAllWindows()
camera.stop()

# Destroy all sensors and vehicles created
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
