"""
CARLA LKA and ACC Testbed with ProPILOT-style State Machine

Authors: Adithya Govindarajan, Pradeepa Hari

- Manual Control: Arrow keys or Logitech G29 Steering Wheel
- ProPILOT Switch: Hold 'P' key for 1.5s (OFF <-> STANDBY)
- Set/Decrease Speed: 'DOWN ARROW'
- Resume/Increase Speed: 'UP ARROW'
- Cancel: 'X' key or Brake Pedal
- Next Scenario: 'N' key
"""

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import carla
import weakref
import random
import numpy as np
import pygame
import argparse
from enum import Enum
import math

# ==============================================================================
# -- ADAS State Machine & Pre-sim GUI ------------------------------------------
# ==============================================================================

class ADAS_State(Enum):
    OFF = 0
    STANDBY = 1
    ACTIVE = 2
    HANDS_OFF = 3

class SettingsMenu:
    """A GUI menu to configure the simulation before it starts."""
    def __init__(self, client, width, height):
        self.client = client
        self.width, self.height = width, height
        self.font = pygame.font.Font(pygame.font.get_default_font(), 22)
        self.title_font = pygame.font.Font(pygame.font.get_default_font(), 48)
        self.vehicles = []
        self.weather_options = [('Clear', carla.WeatherParameters.ClearNoon), ('Light Rain', carla.WeatherParameters.SoftRainNoon), ('Heavy Rain', carla.WeatherParameters.HardRainNoon), ('Foggy', carla.WeatherParameters(fog_density=70.0))]
        
        self.selected_player_vehicle = 'vehicle.tesla.model3'
        self.selected_lead_vehicle = 'vehicle.audi.etron'
        self.selected_weather_index = 0
        
        self.player_scroll_offset = 0
        self.lead_scroll_offset = 0
        
        self._fetch_vehicles()

    def _fetch_vehicles(self):
        self.vehicles = sorted([bp.id for bp in self.client.get_world().get_blueprint_library().filter('vehicle.*')])
        if 'vehicle.tesla.model3' not in self.vehicles and self.vehicles: self.selected_player_vehicle = self.vehicles[0]
        if 'vehicle.audi.etron' not in self.vehicles and self.vehicles: self.selected_lead_vehicle = self.vehicles[0]

    def run(self, display):
        """Runs the main loop for the settings menu."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        if self.start_button_rect.collidepoint(event.pos):
                            return self._get_selected_settings()
                        self._handle_click(event.pos)
                    elif event.button == 4: # Scroll up
                        if event.pos[0] < self.width / 2: self.player_scroll_offset = max(0, self.player_scroll_offset - 1)
                        else: self.lead_scroll_offset = max(0, self.lead_scroll_offset - 1)
                    elif event.button == 5: # Scroll down
                        if event.pos[0] < self.width / 2: self.player_scroll_offset = min(len(self.vehicles) - 10, self.player_scroll_offset + 1)
                        else: self.lead_scroll_offset = min(len(self.vehicles) - 10, self.lead_scroll_offset + 1)

            self._draw(display)
            pygame.display.flip()
    
    def _handle_click(self, pos):
        for i, vehicle_name in enumerate(self.vehicles[self.player_scroll_offset:]):
            rect = pygame.Rect(50, 150 + i * 30, self.width/2 - 100, 30)
            if rect.top > self.height - 150: break
            if rect.collidepoint(pos): self.selected_player_vehicle = vehicle_name; return
        
        for i, vehicle_name in enumerate(self.vehicles[self.lead_scroll_offset:]):
            rect = pygame.Rect(self.width/2 + 50, 150 + i * 30, self.width/2 - 100, 30)
            if rect.top > self.height - 150: break
            if rect.collidepoint(pos): self.selected_lead_vehicle = vehicle_name; return
        
        for i, (name, _) in enumerate(self.weather_options):
            rect = pygame.Rect(50, self.height - 100 + i * 30, 200, 30)
            if rect.collidepoint(pos): self.selected_weather_index = i; return

    def _get_selected_settings(self):
        return {
            'player_filter': self.selected_player_vehicle,
            'lead_filter': self.selected_lead_vehicle,
            'weather': self.weather_options[self.selected_weather_index][1]
        }

    def _draw(self, display):
        display.fill((13, 17, 23))
        title_surf = self.title_font.render("University of Michigan ADAS simulator", True, (255, 203, 5))
        display.blit(title_surf, (self.width/2 - title_surf.get_width()/2, 30))
        player_title = self.font.render("Player Vehicle", True, (0, 127, 255)); display.blit(player_title, (50, 100))
        lead_title = self.font.render("Lead Vehicle", True, (0, 127, 255)); display.blit(lead_title, (self.width/2 + 50, 100))
        for i in range(15):
            idx = self.player_scroll_offset + i
            if idx >= len(self.vehicles): break
            color = (0, 245, 212) if self.vehicles[idx] == self.selected_player_vehicle else (201, 209, 217)
            display.blit(self.font.render(self.vehicles[idx].split('.')[-1], True, color), (50, 150 + i * 30))
        for i in range(15):
            idx = self.lead_scroll_offset + i
            if idx >= len(self.vehicles): break
            color = (0, 245, 212) if self.vehicles[idx] == self.selected_lead_vehicle else (201, 209, 217)
            display.blit(self.font.render(self.vehicles[idx].split('.')[-1], True, color), (self.width/2 + 50, 150 + i * 30))
        display.blit(self.font.render("Initial Weather", True, (0, 127, 255)), (50, self.height - 140))
        for i, (name, _) in enumerate(self.weather_options):
            color = (0, 245, 212) if i == self.selected_weather_index else (201, 209, 217)
            display.blit(self.font.render(name, True, color), (50, self.height - 100 + i*30))
        
        self.start_button_rect = pygame.Rect(self.width/2 - 150, self.height - 80, 300, 50)
        button_color = (0, 80, 150) if self.start_button_rect.collidepoint(pygame.mouse.get_pos()) else (0, 79, 255)
        pygame.draw.rect(display, button_color, self.start_button_rect, border_radius=10)
        display.blit(self.font.render("Start Simulation", True, (255, 255, 255)), (self.width/2 - self.font.render("Start Simulation", True, (255,255,255)).get_width()/2, self.height - 70))

# ==============================================================================
# -- DualControl ---------------------------------------------------------------
# ==============================================================================
class DualControl(object):
    """Manages manual vehicle control inputs from keyboard or steering wheel."""
    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._joystick = None
        if pygame.joystick.get_count() > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            print("Detected Joystick: %s" % self._joystick.get_name())

    def parse_input(self, keys, milliseconds):
        if self._joystick:
            self._parse_wheel()
        else:
            self._parse_keys(keys, milliseconds)
        return self._control

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[pygame.K_UP] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[pygame.K_LEFT]:
            self._steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[pygame.K_DOWN] else 0.0
        self._control.hand_brake = keys[pygame.K_SPACE]

    def _parse_wheel(self):
        numAxes = self._joystick.get_numaxes()
        ButtonKeys = self._joystick.get_numbuttons()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButton = [float(self._joystick.get_button(i)) for i in range(ButtonKeys)]
        
        steerCmd = 1.0 * math.tan(1.1 * jsInputs[0])
        throttleCmd = 0.0
        if jsInputs[1] < 0.9:
            throttleCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[1] + 1.4) - 1.2) / 0.92
            if throttleCmd < 0: throttleCmd = 0.0
            elif throttleCmd > 1: throttleCmd = 1.0

        brakeCmd = 0.0
        if jsInputs[2] < 0.9:
            brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[2] + 1.4) - 1.2) / 0.92
            if brakeCmd < 0: brakeCmd = 0.0
            elif brakeCmd > 1: brakeCmd = 1.0
        
        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

# ==============================================================================
# -- Other Classes and Functions -----------------------------------------------
# ==============================================================================
class ACCController:
    def __init__(self, Kp=0.2, Ki=0.0, Kd=0.1, dt=0.05):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self._prev_error, self._integral = 0.0, 0.0
    def run_step(self, target_speed_ms, current_speed_ms):
        error = target_speed_ms - current_speed_ms
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return output
class StanleyController:
    def __init__(self, k=2.5, k_soft=1.0): self.k, self.k_soft = k, k_soft # Increased gain for centering
    def run_step(self, vehicle, waypoints):
        vehicle_transform = vehicle.get_transform()
        front_axle_location = vehicle_transform.location + vehicle_transform.get_forward_vector() * 1.5
        min_dist, closest_wp = float('inf'), None
        for wp in waypoints:
            dist = front_axle_location.distance(wp.transform.location)
            if dist < min_dist: min_dist, closest_wp = dist, wp
        if closest_wp is None: return 0.0
        wp_transform = closest_wp.transform
        path_heading_rad, vehicle_heading_rad = np.radians(wp_transform.rotation.yaw), np.radians(vehicle_transform.rotation.yaw)
        vec_to_front_axle = front_axle_location - wp_transform.location
        cte = np.dot(np.array([vec_to_front_axle.x, vec_to_front_axle.y]), np.array([-np.sin(path_heading_rad), np.cos(path_heading_rad)]))
        heading_error = path_heading_rad - vehicle_heading_rad
        if heading_error > np.pi: heading_error -= 2 * np.pi
        if heading_error < -np.pi: heading_error += 2 * np.pi
        v_mps = np.linalg.norm([vehicle.get_velocity().x, vehicle.get_velocity().y])
        dynamic_k = self.k / (1.0 + v_mps * 0.5) # Increased damping factor for high-speed stability
        return np.clip(heading_error + np.arctan2(dynamic_k * cte, self.k_soft + v_mps), -1.0, 1.0)
class ScenarioManager:
    def __init__(self, world_obj):
        self.world, self.scenarios = world_obj, [("Baseline", self.setup_baseline), ("Degraded Lanes", self.setup_degraded), ("Cut-in Vehicle", self.setup_cut_in), ("Stop-and-Go", self.setup_stop_go)]
        self.current_scenario_index, self.scenario_actors = -1, []
        self.stop_go_state, self.stop_go_timer = None, 0
        self.lead_vehicle, self.cut_in_vehicle, self.cut_in_state = None, None, None
        self.npc_controller = StanleyController(k=0.7)
        self.next_scenario()
    def cleanup(self):
        for actor_dict in self.scenario_actors:
            actor = actor_dict.get('actor')
            if actor and actor.is_alive: actor.destroy()
        self.scenario_actors.clear()
        self.stop_go_state = self.lead_vehicle = self.cut_in_vehicle = self.cut_in_state = None
    def next_scenario(self):
        self.cleanup()
        self.current_scenario_index = (self.current_scenario_index + 1) % len(self.scenarios)
        name, setup_fn = self.scenarios[self.current_scenario_index]
        setup_fn(); print("Scenario: %s" % name)
    def get_current_scenario_name(self): return self.scenarios[self.current_scenario_index][0]
    def _spawn_scenario_vehicle(self, blueprint, transform):
        vehicle = self.world.world.try_spawn_actor(blueprint, transform)
        if vehicle is None: return None
        self.world.world.tick()
        path, current_wp = [], self.world.map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        for _ in range(200): # Generate a 400m path for the NPC
            next_wps = current_wp.next(2.0)
            if next_wps: current_wp = next_wps[0]; path.append(current_wp)
            else: break
        actor_dict = {'actor': vehicle, 'path': path, 'state': 'DRIVING'}
        self.scenario_actors.append(actor_dict)
        return actor_dict
    def setup_baseline(self): self.world.player.set_transform(self.world.map.get_spawn_points()[2])
    def setup_degraded(self):
        for sp in self.world.map.get_spawn_points():
            if self.world.map.get_waypoint(sp.location).is_intersection: self.world.player.set_transform(sp); return
    def setup_cut_in(self):
        bp_lib, player_wp = self.world.world.get_blueprint_library(), self.world.map.get_waypoint(self.world.player.get_location())
        if player_wp.next(40.0):
            spawn_transform = player_wp.next(40.0)[0].transform; spawn_transform.location.z += 1
            vehicle_dict = self._spawn_scenario_vehicle(bp_lib.find(self.world.args.lead_filter), spawn_transform)
            if vehicle_dict: self.lead_vehicle = vehicle_dict['actor']
        
        left_lane = player_wp.get_left_lane()
        if left_lane and left_lane.next(15.0):
            spawn_transform = left_lane.next(15.0)[0].transform; spawn_transform.location.z += 1
            vehicle_dict = self._spawn_scenario_vehicle(bp_lib.find(self.world.args.lead_filter), spawn_transform)
            if vehicle_dict: self.cut_in_vehicle = vehicle_dict['actor']; vehicle_dict['state'] = "OVERTAKING"
    def setup_stop_go(self):
        player_wp = self.world.map.get_waypoint(self.world.player.get_location())
        if player_wp.next(25.0):
            spawn_transform = player_wp.next(25.0)[0].transform; spawn_transform.location.z += 1
            vehicle_dict = self._spawn_scenario_vehicle(self.world.world.get_blueprint_library().find(self.world.args.lead_filter), spawn_transform)
            if vehicle_dict:
                self.lead_vehicle = vehicle_dict['actor']
                vehicle_dict['state'] = "DRIVING_STOP_GO"; vehicle_dict['start_time'] = pygame.time.get_ticks()
    def tick(self):
        for vehicle_dict in self.scenario_actors:
            vehicle = vehicle_dict.get('actor');
            if not vehicle or not vehicle.is_alive: continue
            
            state, path = vehicle_dict.get('state'), vehicle_dict.get('path')
            steer = self.npc_controller.run_step(vehicle, path) if path else 0.0
            throttle, brake = 0.6, 0.0

            if self.current_scenario_index == 2: # Cut-in
                if vehicle is self.cut_in_vehicle and state == "OVERTAKING":
                    throttle = 0.8
                    if (vehicle.get_location() - self.world.player.get_location()).dot(self.world.player.get_transform().get_forward_vector()) > 10: vehicle_dict['state'] = "CUTTING_IN"
                elif vehicle is self.cut_in_vehicle and state == "CUTTING_IN":
                    steer, throttle = 0.2, 0.6
                    if self.world.map.get_waypoint(vehicle.get_location()).lane_id == self.world.map.get_waypoint(self.world.player.get_location()).lane_id: vehicle_dict['state'] = "MERGED"
                elif vehicle is self.cut_in_vehicle and state == "MERGED":
                    steer, throttle = 0.0, 0.6
            
            elif self.current_scenario_index == 3: # Stop-and-Go
                elapsed_time = pygame.time.get_ticks() - vehicle_dict.get('start_time', 0)
                if state == "DRIVING_STOP_GO":
                    throttle = 0.5
                    if elapsed_time > 4000: vehicle_dict['state'], vehicle_dict['start_time'] = "STOPPING", pygame.time.get_ticks()
                elif state == "STOPPING":
                    throttle, brake = 0.0, 1.0
                    if vehicle.get_velocity().length() < 0.1: vehicle_dict['state'], vehicle_dict['start_time'] = "WAITING", pygame.time.get_ticks()
                elif state == "WAITING":
                    throttle, brake = 0.0, 1.0
                    if pygame.time.get_ticks() - vehicle_dict.get('start_time', 0) > 3000: vehicle_dict['state'], vehicle_dict['start_time'] = "DRIVING_STOP_GO", pygame.time.get_ticks()
            
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

class World:
    def __init__(self, carla_world, args):
        self.world, self.args = carla_world, args
        self.player, self.camera_sensor, self.surface = None, None, None
        try:
            self.map = self.world.get_map()
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(args.player_filter)
            spawn_point = random.choice(self.map.get_spawn_points())
            self.player = self.world.spawn_actor(vehicle_bp, spawn_point)
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(args.width)); camera_bp.set_attribute('image_size_y', str(args.height))
            camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
            weak_self = weakref.ref(self)
            self.camera_sensor.listen(lambda image: World._parse_image(weak_self, image))
        except Exception as e: self.destroy(); raise e
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self: return
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.surface = pygame.surfarray.make_surface(array[:, :, :3][:, :, ::-1].swapaxes(0, 1)) # BGRA → RGB
    def render(self, display):
        if self.surface: display.blit(self.surface, (0, 0))
    def destroy(self):
        if self.camera_sensor: self.camera_sensor.destroy()
        if self.player: self.player.destroy()

def lane_check(world):
    wp = world.world.get_map().get_waypoint(world.player.get_location())
    return wp.left_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.Broken] or \
           wp.right_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.Broken]
def get_upcoming_curvature(world, player, lookahead=30):
    waypoints, current_wp = [], world.get_map().get_waypoint(player.get_location())
    for _ in range(lookahead):
        next_wps = current_wp.next(2.0)
        if next_wps: current_wp = next_wps[0]; waypoints.append(current_wp)
        else: break
    if len(waypoints) < 3: return 0.0
    vectors = [np.array([w.transform.location.x - waypoints[i-1].transform.location.x, w.transform.location.y - waypoints[i-1].transform.location.y]) for i, w in enumerate(waypoints) if i > 0]
    max_curvature = 0
    for i in range(len(vectors) - 1):
        v1_mag, v2_mag = np.linalg.norm(vectors[i]), np.linalg.norm(vectors[i+1])
        if v1_mag > 0 and v2_mag > 0:
            angle = math.acos(np.clip(np.dot(vectors[i], vectors[i+1]) / (v1_mag * v2_mag), -1.0, 1.0))
            curvature = abs(math.degrees(angle))
            if curvature > max_curvature: max_curvature = curvature
    return max_curvature
def get_target_speed_from_curvature(curvature, user_set_speed_kph):
    return user_set_speed_kph * (1.0 - (1.0 / (1.0 + math.exp(-0.2 * (curvature - 5))))) + 25.0 * (1.0 / (1.0 + math.exp(-0.2 * (curvature - 5))))
def get_lead_vehicle(player, carla_world):
    player_transform, player_forward = player.get_transform(), player.get_transform().get_forward_vector()
    min_dist, lead_vehicle = 100.0, None
    for vehicle in carla_world.get_actors().filter('vehicle.*'):
        if vehicle.id != player.id and (vehicle.get_location() - player_transform.location).dot(player_forward) > 0:
            dist = player_transform.location.distance(vehicle.get_location())
            if dist < min_dist: min_dist, lead_vehicle = dist, vehicle
    return lead_vehicle

def game_loop(args, client):
    world, scenario_manager, warning_text, warning_end_time = None, None, None, 0
    try:
        display = pygame.display.get_surface()
        font = pygame.font.Font(pygame.font.get_default_font(), 28)
        world = World(client.get_world(), args)
        world.world.set_weather(args.weather)
        scenario_manager = ScenarioManager(world)

        controller = DualControl()
        stanley, acc_pid = StanleyController(), ACCController()
        adas_state = ADAS_State.OFF
        target_speed_kph, last_set_speed_kph = 0, 35.0
        p_key_press_time, propilot_toggled_this_press = None, False
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            keys = pygame.key.get_pressed()
            current_speed_kph = np.linalg.norm([world.player.get_velocity().x, world.player.get_velocity().y, world.player.get_velocity().z]) * 3.6

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        if p_key_press_time is None: p_key_press_time = pygame.time.get_ticks()

                
                
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        if p_key_press_time is None: p_key_press_time = pygame.time.get_ticks()
                    if event.key == pygame.K_DOWN and adas_state in [ADAS_State.STANDBY]:
                        target_speed_kph, adas_state = current_speed_kph, ADAS_State.ACTIVE
                    if event.key == pygame.K_UP and adas_state == ADAS_State.STANDBY:
                        target_speed_kph, adas_state = last_set_speed_kph, ADAS_State.ACTIVE
                    if event.key == pygame.K_x:
                        last_set_speed_kph, adas_state = target_speed_kph, ADAS_State.STANDBY
                    if event.key == pygame.K_n: scenario_manager.next_scenario()
                if event.type == pygame.KEYUP and event.key == pygame.K_p:
                    p_key_press_time, propilot_toggled_this_press = None, False
            
            if p_key_press_time is not None and not propilot_toggled_this_press and pygame.time.get_ticks() - p_key_press_time > 1500:
                adas_state = ADAS_State.STANDBY if adas_state == ADAS_State.OFF else ADAS_State.OFF
                propilot_toggled_this_press = True

            scenario_manager.tick()

            manual_control = controller.parse_input(keys, clock.get_time())
            if manual_control.brake > 0.1 and adas_state not in [ADAS_State.OFF, ADAS_State.STANDBY]:
                last_set_speed_kph, adas_state = target_speed_kph, ADAS_State.STANDBY
                warning_text, warning_end_time = "System Standby: Brake Applied", pygame.time.get_ticks() + 2000

            upcoming_curvature = get_upcoming_curvature(world.world, world.player)
            cornering_threshold = 1.5 
            if adas_state == ADAS_State.ACTIVE and lane_check(world) and upcoming_curvature < cornering_threshold:
                adas_state = ADAS_State.HANDS_OFF
            elif adas_state == ADAS_State.HANDS_OFF and (not lane_check(world) or upcoming_curvature >= cornering_threshold):
                adas_state = ADAS_State.ACTIVE
            
            if adas_state in [ADAS_State.ACTIVE, ADAS_State.HANDS_OFF] and current_speed_kph < 0.5:
                adas_state, last_set_speed_kph = ADAS_State.STANDBY, target_speed_kph

            throttle, brake, steer = manual_control.throttle, manual_control.brake, manual_control.steer
            
            if adas_state in [ADAS_State.ACTIVE, ADAS_State.HANDS_OFF]:
                final_target_kph = get_target_speed_from_curvature(upcoming_curvature, target_speed_kph)
                lead_vehicle = get_lead_vehicle(world.player, world.world)
                if lead_vehicle:
                    final_target_kph = min(final_target_kph, (lead_vehicle.get_velocity().length() * 3.6) + ((world.player.get_location().distance(lead_vehicle.get_location()) - 10.0) * 0.5))
                control_signal = acc_pid.run_step(final_target_kph / 3.6, current_speed_kph / 3.6)
                throttle, brake = (max(0, control_signal), max(0, -control_signal))

            if adas_state == ADAS_State.HANDS_OFF:
                waypoints = [world.world.get_map().get_waypoint(world.player.get_location())]
                for _ in range(50):
                    next_wps = waypoints[-1].next(2.0)
                    if next_wps: waypoints.append(next_wps[0])
                    else: break
                steer = stanley.run_step(world.player, waypoints)
            
            world.player.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            
            world.render(display)
            draw_adas_hud(display, font, adas_state, current_speed_kph, target_speed_kph, warning_text, warning_end_time, scenario_manager)
            pygame.display.flip()
    finally:
        if world: world.destroy()
        if scenario_manager: scenario_manager.cleanup()

def draw_adas_hud(display, font, state, current_speed, target_speed, warning_text, warning_end_time, scenario_manager):
    width = display.get_width()
    state_text = state.name.replace('_', ' ')
    state_color = {ADAS_State.OFF: (150,150,150), ADAS_State.STANDBY: (150,150,150), ADAS_State.ACTIVE: (0,100,255), ADAS_State.HANDS_OFF: (0,200,0)}.get(state, (255,0,0))
    
    display.blit(font.render("SCENARIO: %s" % scenario_manager.get_current_scenario_name(), True, (255,255,255)), (20, 20))
    
    text_surface = font.render(state_text, True, state_color)
    text_rect = text_surface.get_rect(topright=(width - 20, 20))
    display.blit(text_surface, text_rect)

    big_font = pygame.font.Font(pygame.font.get_default_font(), 64)
    speed_surface = big_font.render(f"{int(current_speed)}", True, (255, 255, 255))
    speed_rect = speed_surface.get_rect(topright=(width - 30, 60))
    display.blit(speed_surface, speed_rect)
    
    if state in [ADAS_State.ACTIVE, ADAS_State.HANDS_OFF]:
        target_surface = font.render(f"SET: {int(target_speed)}", True, (200, 200, 200))
        target_rect = target_surface.get_rect(topright=(width - 25, 125))
        display.blit(target_surface, target_rect)
        
    if warning_text and pygame.time.get_ticks() < warning_end_time:
        warn_font = pygame.font.Font(pygame.font.get_default_font(), 32)
        warn_surface = warn_font.render(warning_text, True, (255, 0, 0))
        warn_rect = warn_surface.get_rect(center=(width/2, 50))
        display.blit(warn_surface, warn_rect)

def main():
    argparser = argparse.ArgumentParser(description='CARLA LKA & ACC Testbed')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--width', default=1280, type=int, help='Window width')
    argparser.add_argument('--height', default=720, type=int, help='Window height')
    argparser.add_argument('--filter', default='vehicle.*', help='Player vehicle filter')
    argparser.add_argument('--map', default='Town10HD_Opt', help='Map to load')
    args = argparser.parse_args()
    try:
        pygame.init()
        pygame.font.init()
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        print("Loading map: %s..." % args.map)
        client.load_world(args.map)
        print("Map loaded.")
        
        menu = SettingsMenu(client, args.width, args.height)
        settings = menu.run(display)

        if settings:
            args.player_filter = settings['player_filter']
            args.lead_filter = settings['lead_filter']
            args.weather = settings['weather']
            game_loop(args, client)
        
    except Exception as e:
        print("\nAn error occurred: %s" % e)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
