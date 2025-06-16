



# CARLA ADAS Testbed: LKA & ACC with ProPILOT-style Logic



This project is an Advanced Driver-Assistance Systems (ADAS) testbed developed in Python using the [CARLA Simulator](http://carla.org/). It features a robust implementation of Lane Keep Assist (LKA) and Adaptive Cruise Control (ACC), governed by a state machine inspired by real-world systems like Nissan's ProPILOT.

The simulation allows for testing the ADAS controllers under various configurable scenarios, including vehicle cut-ins, stop-and-go traffic, and degraded lane markings. It includes a pre-simulation GUI for selecting vehicles and weather conditions, and provides a clear on-screen HUD for monitoring the ADAS state.

This simulator was developed by **Adithya Govindarajan** and **Pradeepa Hari**.

## Features

  - **ProPILOT-style State Machine:** A multi-state system (OFF, STANDBY, ACTIVE, HANDS-OFF) for intuitive and safe control of the ADAS features.
  - **Lane Keep Assist (LKA):** Utilizes a **Stanley Controller** for smooth and accurate lane centering. The system intelligently deactivates on sharp curves or where lane markings are not detected.
  - **Adaptive Cruise Control (ACC):** Implemented with a **PID Controller** to maintain a set speed and safely follow lead vehicles. It dynamically adjusts speed based on upcoming road curvature.
  - **Dynamic Scenarios:** A built-in `ScenarioManager` allows cycling through different test cases on-the-fly:
      - **Baseline:** Standard highway driving.
      - **Degraded Lanes:** Tests LKA robustness in areas with poor lane markings (e.g., intersections).
      - **Cut-in Vehicle:** An NPC vehicle from an adjacent lane merges in front of the player.
      - **Stop-and-Go:** A lead vehicle simulates traffic by accelerating, braking to a full stop, and resuming.
  - **Pre-Simulation GUI:** A user-friendly menu created with Pygame to select the player vehicle, lead vehicle, and initial weather conditions.
  - **Dual Control Support:** Full support for manual driving using either a keyboard or a Logitech G29 steering wheel for a more immersive experience.
  - **Informative HUD:** A clean heads-up display shows the current ADAS state, vehicle speed, set speed, and critical system warnings.

## System Architecture

The controller is designed around a state machine that dictates the level of autonomous control:

  - **`OFF`**: No ADAS features are active. The user has full manual control.
  - **`STANDBY`**: The system is armed but not controlling the vehicle. The user can activate ACC/LKA from this state.
  - **`ACTIVE`**: ACC is engaged, managing speed and following distance. The user is responsible for steering.
  - **`HANDS_OFF`**: Both ACC and LKA are engaged. The system manages speed, following distance, and steering. The system will transition back to `ACTIVE` if lane markings are lost or a sharp curve is detected.

Lateral control (LKA) is handled by the **Stanley Controller**, which calculates the optimal steering angle based on heading error and cross-track error relative to the lane's center path.

Longitudinal control (ACC) is managed by a **PID Controller**, which modulates the throttle and brake to match a target speed. This target speed is the lower of the user's set speed or a dynamically calculated speed based on lead vehicle proximity and road curvature.

## Prerequisites

  - **CARLA Simulator (0.9.13 or newer recommended):** Follow the official [CARLA documentation](https://carla.readthedocs.io/en/latest/getting_started/) to install it.
  - **Python 3.7+**
  - **Python packages:**
      - `pygame`
      - `numpy`
      - `carla`

## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/carla-adas-testbed.git
    cd carla-adas-testbed
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install pygame numpy carla
    ```

3.  **Launch the CARLA Server:**
    Navigate to your CARLA installation directory and run the server.

    ```bash
    # For Windows
    CarlaUE4.exe -prefernvidia

    # For Linux
    ./CarlaUE4.sh
    ```

    Optionally, you can specify a map that matches the one in the script (`Town10HD_Opt` by default).

    ```bash
    ./CarlaUE4.sh -map=Town10HD_Opt
    ```

## Running the Simulation

With the CARLA server running, execute the main Python script:

```bash
python Propilot_simulator.py
```

You can also use command-line arguments to configure the simulation:

```bash
python Propilot_simulator.py --host 127.0.0.1 --port 2000 --width 1280 --height 720
```

## Controls

| Action                  | Keyboard Control              | Logitech G29 Control |
| ----------------------- | ----------------------------- | -------------------- |
| **Steering** | `Left/Right Arrow Keys`       | Steering Wheel       |
| **Throttle / Brake** | `Up/Down Arrow Keys`          | Accelerator / Brake Pedals |
| **Toggle ProPILOT** | Hold `P` key for 1.5s         | *Not mapped* |
| **Set / Decrease Speed**| `Down Arrow` (in STANDBY)     | *Not mapped* |
| **Resume / Increase Speed**| `Up Arrow` (in STANDBY)    | *Not mapped* |
| **Cancel ADAS** | `X` key                       | Brake Pedal          |
| **Next Scenario** | `N` key                       | *Not mapped* |
| **Exit** | `ESC` key                     | -                    |

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
