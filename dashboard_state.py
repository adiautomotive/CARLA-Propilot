from enum import Enum

class ADAS_State(Enum):
    OFF = 0
    STANDBY = 1
    ACTIVE = 2
    HANDS_OFF = 3
    
class VehicleState:
    def __init__(self):
        self.current_speed_kph = 0.0
        self.target_speed_kph = 0.0
        self.last_set_speed_kph = 0.0
        self.gear = "P"
        self.rpm = 0
        self.adas_state = ADAS_State.OFF