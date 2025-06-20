import carla
import math
from agents.navigation.global_route_planner import GlobalRoutePlanner

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
