import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import pygame
from PIL import Image, ImageDraw, ImageTk
# import start_vehicle_2 as simulation_main  # your simulation script
import simulation_main  # your simulation script
import random
import time
from datetime import datetime
import requests
API_KEY = "f10450bfaf141d03cf1860e45a8e9e06"
CITY = "Michigan"
UNITS = 'imperial'  # use 'imperial' for Fahrenheit 'metric'
temperature = "46"
def fetch_temp():
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200 and "main" in data:
        return data['main']['temp']
    else:
        return temperature

class ADASSimulatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("University of Michigan ADAS Simulator")
        self.geometry("1280x720")

        pygame.init()
        pygame.joystick.init()

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartMenu, DashboardScreen):
            frame = F(parent=self.container, controller=self)
            self.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame(StartMenu)

    def show_frame(self, context):
        frame = self.frames[context]
        frame.tkraise()

    def start_simulation_with_params(self, params):
        threading.Thread(target=self.run_simulation, args=(params,), daemon=True).start()
        self.show_frame(DashboardScreen)
        # self.frames[DashboardScreen].start_dashboard_update()

    def run_simulation(self, params):
        try:
            simulation_main.main(params)
        except Exception as e:
            print(f"[ERROR] {e}")


class StartMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # === Background Image ===
        bg_image = Image.open("/home/jesudara/carla_dev/carla/ADASProject/dev/ADAS_Simulator/images/Meeting-Critical.jpg")
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        tk.Label(self, image=self.bg_photo).place(relwidth=1, relheight=1)

        self.widgets = []
        self.init_styles()
        self.create_widgets()

        self.focus_index = 0
        self.set_focus(self.focus_index)
        self.bind_all_keys()

        self.init_joystick()

    def init_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#00274C", foreground="white")
        style.configure("TCheckbutton", background="#00274C", foreground="white")
        style.configure("TButton", background="#2f4f6f", foreground="white", relief="flat")
        style.configure("TCombobox", fieldbackground="white", background="white", foreground="black")
        style.configure("Start.TButton", font=("Helvetica", 14, "bold"), padding=(10, 8))


    def create_widgets(self):
        left_x, right_x = 0.25, 0.65
        top_y, step_y = 0.15, 0.07
        font_inline = font.Font(size=14)

        # Variables
        self.vehicle_model = tk.StringVar(value="Tesla Model 3")
        self.weather = tk.StringVar(value="Clear Noon")
        self.scenario = tk.StringVar(value="Highway Merge")
        self.lead_vehicle = tk.BooleanVar()
        self.traffic_vehicles = tk.IntVar(value=10)
        self.pedestrians = tk.IntVar(value=5)

        # ttk.Label(self, text="University of Michigan ADAS Simulator", font=("Helvetica", 18, "bold"), bg="#00274C", fg='white').place(relx=0.5, rely=0.05, anchor='center')
        tk.Label(self, text="University of Michigan ADAS Simulator", font=("Helvetica", 28, "bold"), bg="#00274C", fg='white').place(relx=0.5, rely=0.05, anchor='center')


        inputs = [
            (left_x, "Player Vehicle:", self.vehicle_model, ["Tesla Model 3", "Lincoln MKZ", "Audi A2"]),
            (right_x, "Initial Weather:", self.weather, ["Clear Noon", "Cloudy Noon", "Wet Noon", "Mid Rain", "Hard Rain", "Soft Rain", "Clear Sunset"]),
            (right_x, "Initial Scenario:", self.scenario, ["Highway Merge", "City Stop-Go", "Rural Cruise"]),
        ]

        for i, (x, label, var, options) in enumerate(inputs):
            ttk.Label(self, text=label, font=font_inline).place(relx=x, rely=top_y + i*step_y, anchor='center')
            cb = ttk.Combobox(self, textvariable=var, values=options, state="readonly")
            cb.current(0)
            cb.place(relx=x, rely=top_y + i*step_y + 0.03, anchor='center')
            self.widgets.append(cb)

        # Lead Vehicle Check
        ttk.Label(self, text="Lead Vehicle:", font=font_inline).place(relx=right_x, rely=top_y + 3 * step_y, anchor='center')
        cb = ttk.Checkbutton(self, text="Enable", variable=self.lead_vehicle)
        cb.place(relx=right_x, rely=top_y + 3 * step_y + 0.03, anchor='center')
        self.widgets.append(cb)

        # Entry Fields
        entry_fields = [
            ("Number of Traffic Vehicles:", self.traffic_vehicles),
            ("Number of Pedestrians:", self.pedestrians),
        ]
        for i, (label, var) in enumerate(entry_fields, start=4):
            ttk.Label(self, text=label, font=font_inline).place(relx=right_x, rely=top_y + i * step_y, anchor='center')
            entry = ttk.Entry(self, textvariable=var)
            entry.place(relx=right_x, rely=top_y + i * step_y + 0.03, anchor='center')
            self.widgets.append(entry)

        # Log Box
        ttk.Label(self, text="Log:", font=font_inline).place(relx=0.23, rely=0.68, anchor='w')
        self.log_box = tk.Text(self, height=3, width=95, state='disabled', bg="#f0f0f0", fg="black")
        self.log_box.place(relx=0.5, rely=0.73, anchor='center')

        # Start Button
        # self.start_button = ttk.Button(self, text="Start Simulation", command=self.start_simulation)
        self.start_button = ttk.Button(self, text="Start Simulation", style="Start.TButton", command=self.start_simulation)
        self.start_button.place(relx=0.5, rely=0.92, anchor='center')
        self.widgets.append(self.start_button)

    def bind_all_keys(self):
        self.bind("<Down>", self.next_focus)
        self.bind("<Up>", self.prev_focus)
        self.bind("<Return>", self.activate_focused)
        self.bind("<Left>", self.modify_value)
        self.bind("<Right>", self.modify_value)

    def init_joystick(self):
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.after(100, self.poll_joystick)
        else:
            self.joystick = None

    # def start_simulation(self):
    #     params = {
    #         "vehicle_model": self.vehicle_model.get(),
    #         "weather": self.weather.get(),
    #         "scenario": self.scenario.get(),
    #         "lead_vehicle": self.lead_vehicle.get(),
    #         "traffic_vehicles": self.traffic_vehicles.get(),
    #         "pedestrians": self.pedestrians.get(),
    #         "host": "localhost",
    #         "port": 2000
    #     }
    #     self.controller.start_simulation_with_params(params)
    def start_simulation(self):
        params = {
            # "vehicle_model": self.player_vehicle.get(),
            "vehicle_model": self.vehicle_model.get(),
            "weather": self.weather.get(),
            "host": "localhost",
            "port": 2000
        }
        self.controller.start_simulation_with_params(params)

    def set_focus(self, index):
        try:
            self.widgets[index].focus_set()
        except IndexError:
            pass

    def next_focus(self, event=None):
        self.focus_index = (self.focus_index + 1) % len(self.widgets)
        self.set_focus(self.focus_index)

    def prev_focus(self, event=None):
        self.focus_index = (self.focus_index - 1) % len(self.widgets)
        self.set_focus(self.focus_index)

    def activate_focused(self, event=None):
        widget = self.widgets[self.focus_index]
        if isinstance(widget, ttk.Checkbutton):
            self.lead_vehicle.set(not self.lead_vehicle.get())
        elif widget == self.start_button:
            self.start_simulation()

    def modify_value(self, event=None):
        widget = self.widgets[self.focus_index]
        if isinstance(widget, ttk.Combobox):
            current = widget.current()
            values = widget['values']
            widget.current((current + 1) % len(values) if event.keysym == "Right" else (current - 1) % len(values))
        elif isinstance(widget, ttk.Entry):
            try:
                val = int(widget.get())
                val += 1 if event.keysym == "Right" else -1
                val = max(0, val)
                widget.delete(0, tk.END)
                widget.insert(0, str(val))
            except ValueError:
                pass

    def poll_joystick(self):
        if not self.joystick:
            return
        pygame.event.pump()
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() else (0, 0)
        if hat[1] == 1:
            self.prev_focus()
        elif hat[1] == -1:
            self.next_focus()
        elif hat[0] != 0:
            self.modify_value(type('event', (), {'keysym': "Right" if hat[0] > 0 else "Left"}))
        if self.joystick.get_button(0):
            self.activate_focused()
        elif self.joystick.get_button(7):
            self.start_simulation()
        self.after(100, self.poll_joystick)

    def log(self, message):
        self.log_box.configure(state='normal')
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.configure(state='disabled')
        self.log_box.yview(tk.END)


class DashboardScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="black")
        self.controller = controller
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.images = []
        self.canvas.pack(fill="both", expand=True)

        # Dashboard state variables
        self.rpm = 0
        self.speed = 0
        self.target_rpm = 0
        self.target_speed = 0
        self.gear = "P"

        # ProPILOT State
        self.propilot_state = 0  # ranges 0–4
        self.propilot_timer = 0

        # Load fuel icon
        icon_dir="/home/jesudara/carla_dev/carla/ADASProject/dev/ADAS_Simulator/images/"
        fuel_image = Image.open(icon_dir + "icons/gas-pump-512.ico")
        fuel_image = fuel_image.resize((20, 20), Image.LANCZOS)
        self.fuel_icon = ImageTk.PhotoImage(fuel_image)
        car_img = Image.open(icon_dir +"icons/white_car.png").resize((50, 50), Image.LANCZOS)
        self.car_icon = ImageTk.PhotoImage(car_img)

        self.after(100, self.update_dashboard)

    
    def draw_fading_line_on_canvas(self, x, y, length, thickness=4, color=(255, 255, 255), fade_length=50):
        img_width = length
        img_height = thickness
        image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        solid_start = fade_length
        solid_end = length - fade_length
        draw.rectangle([solid_start, 0, solid_end, thickness], fill=color + (255,))


        for i in range(fade_length):
            alpha = int(255 * (i / fade_length))
            draw.line([(i, 0), (i, thickness)], fill=color + (alpha,))
            x_right = solid_end + i
            draw.line([(x_right, 0), (x_right, thickness)], fill=color + (255 - alpha,))

        tk_img = ImageTk.PhotoImage(image)
        self.canvas.create_image(x, y, anchor="nw", image=tk_img)
        self.canvas.images.append(tk_img)  # keep reference

    def draw_road_with_fade(self, start_x, start_y, width, height, lane_count=3, fade_height=100):
        # Create image for road scene
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Road dimensions (relative to image)
        road_bottom_width = width * 0.9
        road_top_width = width * 0.2
        road_top_y = height * 0.2
        road_bottom_y = height

        # Compute points for the trapezoid road
        road_points = [
            ((width - road_top_width) / 2, road_top_y),
            ((width + road_top_width) / 2, road_top_y),
            ((width + road_bottom_width) / 2, road_bottom_y),
            ((width - road_bottom_width) / 2, road_bottom_y),
        ]
        draw.polygon(road_points, fill=(50, 50, 50, 255))
        # draw.polygon(road_points, fill=(0, 0, 0, 255))  # black road

        # === Draw Lane Lines ===
        total_lines = lane_count  # for 3 lanes, draw 4 lines (including dashed at far left)
        for i in range(total_lines+1):
            lane_x_top = (width - road_top_width) / 2 + i * (road_top_width / lane_count)
            lane_x_bottom = (width - road_bottom_width) / 2 + i * (road_bottom_width / lane_count)

            if i == 0:
                # Dashed leftmost lane line
                segments = 20
                for j in range(segments):
                    y1 = road_top_y + j * (road_bottom_y - road_top_y) / segments
                    y2 = y1 + (road_bottom_y - road_top_y) / (2 * segments)
                    x1 = lane_x_top + (lane_x_bottom - lane_x_top) * (j / segments)
                    x2 = x1
                    draw.line([(x1, y1), (x2, y2)], fill=(200, 200, 200, 180), width=2)
            else:
                # Solid white lane line
                draw.line([(lane_x_top, road_top_y), (lane_x_bottom, road_bottom_y)],
                        fill=(255, 255, 255, 200), width=2)

        # === Bottom Fade ===
        for i in range(fade_height):
            alpha = int(255 * (1 - i / fade_height))
            y = height - fade_height + i
            draw.rectangle([0, y, width, y + 1], fill=(0, 0, 0, 255 - alpha))

        # Convert and display the road image
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(start_x, start_y, anchor="nw", image=tk_image)
        self.canvas.images.append(tk_image)  # NEWion

        # === Add horizontal fading line near bottom ===
        w = self.winfo_width()
        h = self.winfo_height()
        cx = w // 2
        cy = h // 2
        line_length = width * 0.8
        line_x = (width - line_length) / 2 + start_x
        line_y = start_y + height - 40  # slightly above the bottom of the road
        self.draw_fading_line_on_canvas(x=cx-160, y=cy+110, length=320, thickness=2, color=(255, 255, 255), fade_length=80)


    def update_dashboard(self):
        self.canvas.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        cx = w // 2
        cy = h // 2
        self.draw_road_with_fade(start_x=cx/2 + 80, start_y=cy/2 + 70, width=480, height=320)

        self.simulate_drive()

        # === Time and Temp
        temp = fetch_temp()
        
        current_time = datetime.now().strftime("%I:%M %p").lstrip("0")
        self.canvas.create_text(cx, cy - 180, text=f"{temp} °F     {current_time}", fill="white", font=("Helvetica", 10))

        # === ProPILOT Assist Mode Indicator
        mode_colors = ["black", "gray", "white", "green", "blue"]
        mode_labels = [
            "System Off", "Standby Mode", "ICC Active",
            "ProPILOT Assist", "Hands-Off Mode"
        ]
        mode_color = mode_colors[self.propilot_state]
        mode_text = mode_labels[self.propilot_state]
        self.canvas.create_text(cx, cy - 140, text=mode_text,
                                fill=mode_color, font=("Helvetica", 18, "bold"))

        # === Draw Dials
        self.draw_dial_style(cx - 250, cy, self.rpm / 8000, "RPMx1000", f"{int(self.rpm)}",
                       label="AUTO", gear=self.gear, max_value=8000, step=1000,
                       label_formatter=lambda v: str(int(v // 1000)))

        self.draw_dial_2(cx + 250, cy, self.speed / 160, "mph", f"{int(self.speed)}",
                       label="LIMIT\n 45", max_value=160, step=20)

        # === Bottom Panel
        self.canvas.create_image(cx + 90, cy + 130, image=self.fuel_icon, anchor="w")
        self.canvas.create_text(cx - 140, cy + 130, text="5612", fill="white", font=("Helvetica", 12, "bold"), anchor="w")
        self.canvas.create_text(cx + 120, cy + 130, text="278", fill="white", font=("Helvetica", 12, "bold"), anchor="w")
        self.canvas.create_text(cx - 140, cy + 145, text="mile", fill="white", font=("Helvetica", 10), anchor="w")
        self.canvas.create_text(cx + 120, cy + 145, text="mile", fill="white", font=("Helvetica", 10), anchor="w")
        # self.draw_fading_line(x1=cx-160, y=cy+110, length=320, thickness=2, color=(255, 255, 255), fade_length=80)
        self.canvas.create_image(cx, (cy + 50), image=self.car_icon)



        self.after(100, self.update_dashboard)

    def simulate_drive(self):
        # Simulate acceleration/deceleration
        if self.target_speed < 80:
            self.target_speed += random.randint(1, 5)
            self.target_rpm = self.target_speed * 80
        else:
            self.target_speed = random.randint(40, 80)
            self.target_rpm = self.target_speed * 80

        # Smooth easing
        self.speed += (self.target_speed - self.speed) * 0.1
        self.rpm += (self.target_rpm - self.rpm) * 0.1

        # Gear simulation
        if self.speed < 1:
            self.gear = "P"
        elif self.speed < 5:
            self.gear = "R"
        elif self.speed < 20:
            self.gear = "N"
        elif self.speed > 30:
            self.gear = "D"

        # === ProPILOT State Machine ===
        self.propilot_timer += 1
        if self.propilot_timer > 50:  # every ~5 seconds
            self.propilot_state = (self.propilot_state + 1) % 5
            self.propilot_timer = 0

    def draw_dial(self, x, y, percent, unit, value, label="", gear=None, max_value=100, step=10, label_formatter=None):
        import math
        r = 120
        start_angle = 225
        sweep_angle = 270
        total_ticks = int(max_value / step)

        tick_outer = r
        tick_inner = r - 10
        text_radius = r + 15

        # Circle Border
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="white", width=2)

        # Tick Marks
        for i in range(total_ticks + 1):
            angle_deg = start_angle - (i / total_ticks) * sweep_angle
            angle_rad = math.radians(angle_deg)
            x_outer = x + tick_outer * math.cos(angle_rad)
            y_outer = y - tick_outer * math.sin(angle_rad)
            x_inner = x + tick_inner * math.cos(angle_rad)
            y_inner = y - tick_inner * math.sin(angle_rad)
            self.canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill="white", width=2)

            tick_val = i * step
            tick_label = label_formatter(tick_val) if label_formatter else str(tick_val)
            x_text = x + text_radius * math.cos(angle_rad)
            y_text = y - text_radius * math.sin(angle_rad)
            self.canvas.create_text(x_text, y_text, text=tick_label, fill="white", font=("Helvetica", 8))

        # Red Arc
        arc_extent = -percent * sweep_angle
        self.canvas.create_arc(x - r, y - r, x + r, y + r,
                               start=start_angle, extent=arc_extent,
                               style="arc", outline="red", width=8)

        # Needle
        needle_angle = start_angle + percent * sweep_angle
        needle_rad = math.radians(needle_angle)
        needle_len = r - 15
        x_end = x + needle_len * math.cos(needle_rad)
        y_end = y - needle_len * math.sin(needle_rad)
        self.canvas.create_line(x, y, x_end, y_end, fill="white", width=3)

        # Dial Center and Texts
        self.canvas.create_oval(x - 90, y - 90, x + 90, y + 90, fill="black", outline="gray")
        # self.canvas.create_text(x, y - 10, text=value, fill="white", font=("Helvetica", 16, "bold"))
        self.canvas.create_text(x, y - 50, text=unit, fill="white", font=("Helvetica", 8, "bold"))

        if label:
            bbox = self.canvas.bbox(self.canvas.create_text(x, y + 50, text=label, fill="gray", font=("Helvetica", 10)))
            # Add padding
            pad = 2
            x1, y1, x2, y2 = bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad

            # Draw rectangle around text
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=2)
        if gear:
            self.canvas.create_text(x, y, text=gear, fill="#00FFAA", font=("Helvetica", 32, "bold"))

    def draw_dial_style(self, x, y, percent, unit, value, label="", gear=None, max_value=100, step=10, label_formatter=None):
        import math
        r = 120
        start_angle = 225
        sweep_angle = 270
        total_ticks = int(max_value / step)
        minor_per_major = 4  # i.e., 5 total divisions per step
        minor_step_deg = sweep_angle / (total_ticks* (minor_per_major + 1))


        tick_outer = r + 15
        tick_inner = r
        text_radius = r - 15

        # Red Arc
        # arc_extent = -percent * sweep_angle
        # self.canvas.create_arc(x - r, y - r, x + r, y + r,
        #                        start=start_angle, extent=arc_extent,
        #                        style="arc", outline="red", width=8)
        
        # === Grey Gradient Fill (inner)
        r_inner = 95
        grey_extent = -percent * sweep_angle
        self.canvas.create_arc(x - r_inner, y - r_inner, x + r_inner, y + r_inner,
                            start=start_angle, extent=grey_extent,
                            style="arc", outline="gray", width=100)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="white",fill="grey", width=1)
        # r_inner= 95

        self.canvas.create_oval(x - r_inner, y - r_inner, x + r_inner, y + r_inner,fill="red", width=2)

        # Circle Border
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="white", width=2)

        # Tick Marks
        for i in range(total_ticks *(minor_per_major + 1) + 1):

            # angle_deg = start_angle - (i / total_ticks) * sweep_angle
            # angle_rad = math.radians(angle_deg)
            angle_deg = start_angle - i * minor_step_deg
            angle_rad = math.radians(angle_deg)


            # Determine if it's a major or minor tick
            is_major = (i % (minor_per_major + 1) == 0)
            length = 10 if is_major else 5
            color = "white" if is_major else "white"
            width = 2 if is_major else 1
            outer = tick_outer + 5
            inner = outer - length
            x_outer = x + outer * math.cos(angle_rad)
            y_outer = y - outer * math.sin(angle_rad)
            x_inner = x + inner * math.cos(angle_rad)
            y_inner = y - inner * math.sin(angle_rad)
            self.canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill=color, width=width)


            # x_outer = x + tick_outer * math.cos(angle_rad)
            # y_outer = y - tick_outer * math.sin(angle_rad)
            # x_inner = x + tick_inner * math.cos(angle_rad)
            # y_inner = y - tick_inner * math.sin(angle_rad)
            # self.canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill="white", width=2)

            # # Draw label only on major ticks
            # if is_major:
            #     tick_val = (i // (total_ticks + 1)) * step
            #     label = label_formatter(tick_val) if label_formatter else str(tick_val)
            #     x_text = x + text_radius * math.cos(angle_rad)
            #     y_text = y - text_radius * math.sin(angle_rad)
            #     self.canvas.create_text(x_text, y_text, text=label, fill="white", font=("Helvetica", 8))
            if is_major:
                tick_val = (i * step)/5
                tick_label = label_formatter(tick_val) if label_formatter else str(tick_val)
                x_text = x + text_radius * math.cos(angle_rad)
                y_text = y - text_radius * math.sin(angle_rad)
                self.canvas.create_text(x_text, y_text, text=tick_label, fill="white", font=("Times New Roman", 10, "bold"))

        

        # Needle
        # needle_angle = start_angle + percent * sweep_angle
        needle_angle = start_angle - percent * sweep_angle
        needle_rad = math.radians(needle_angle)
        needle_len = r + 10
        x_end = x + needle_len * math.cos(needle_rad)
        y_end = y - needle_len * math.sin(needle_rad)
        self.canvas.create_line(x, y, x_end, y_end, fill="red", width=3)

        # Dial Center and Texts
        self.canvas.create_oval(x - 90, y - 90, x + 90, y + 90, fill="black", outline="gray")
        # self.canvas.create_text(x, y - 10, text=value, fill="white", font=("Helvetica", 16, "bold"))
        self.canvas.create_text(x, y - 50, text=unit, fill="white", font=("Helvetica", 8, "bold"))

        if label:
            bbox = self.canvas.bbox(self.canvas.create_text(x, y + 50, text=label, fill="gray", font=("Helvetica", 10)))
            # Add padding
            pad = 2
            x1, y1, x2, y2 = bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad

            # Draw rectangle around text
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=2)
        if gear:
            self.canvas.create_text(x, y, text=gear, fill="#00FFAA", font=("Helvetica", 32, "bold"))

            
    def draw_dial_2(self, x, y, percent, unit, value, label="", gear=None, max_value=100, step=10, label_formatter=None):
        import math
        r = 120
        start_angle = 225
        sweep_angle = 270
        total_ticks = int(max_value / step)
        minor_per_major = 4  # i.e., 5 total divisions per step
        minor_step_deg = sweep_angle / (total_ticks* (minor_per_major + 1))


        tick_outer = r + 15
        tick_inner = r 
        text_radius = r - 15

        # Circle Border
        
        # === Grey Gradient Fill (inner)
        r_inner = 95
        grey_extent = -percent * sweep_angle
        self.canvas.create_arc(x - r_inner, y - r_inner, x + r_inner, y + r_inner,
                            start=start_angle, extent=grey_extent,
                            style="arc", outline="gray", width=100)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="white",fill="grey", width=1)
        # r_inner= 95

        self.canvas.create_oval(x - r_inner, y - r_inner, x + r_inner, y + r_inner,fill="red", width=2)
        

        # Tick Marks
        for i in range(total_ticks *(minor_per_major + 1) + 1):
            # angle_deg = start_angle - (i / total_ticks) * sweep_angle
            # angle_rad = math.radians(angle_deg)
            angle_deg = start_angle - i * minor_step_deg
            angle_rad = math.radians(angle_deg)

            # Determine if it's a major or minor tick
            is_major = (i % (minor_per_major + 1) == 0)
            length = 10 if is_major else 5
            color = "white" if is_major else "white"
            width = 2 if is_major else 1
            outer = tick_outer + 5
            inner = outer - length

            x_outer = x + outer * math.cos(angle_rad)
            y_outer = y - outer * math.sin(angle_rad)
            x_inner = x + inner * math.cos(angle_rad)
            y_inner = y - inner * math.sin(angle_rad)
            self.canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill=color, width=width)


            # x_outer = x + tick_outer * math.cos(angle_rad)
            # y_outer = y - tick_outer * math.sin(angle_rad)
            # x_inner = x + tick_inner * math.cos(angle_rad)
            # y_inner = y - tick_inner * math.sin(angle_rad)
            # self.canvas.create_line(x_inner, y_inner, x_outer, y_outer, fill="white", width=2)

            if is_major:
                tick_val = int((i * step)/5)
                tick_label = label_formatter(tick_val) if label_formatter else str(tick_val)
                x_text = x + text_radius * math.cos(angle_rad)
                y_text = y - text_radius * math.sin(angle_rad)
                self.canvas.create_text(x_text, y_text, text=tick_label, fill="white", font=("Times New Roman", 10, "bold"))
                # self.canvas.lift(text_id2)

        
        
        # Red Arc
        # arc_extent = -percent * sweep_angle
        # self.canvas.create_arc(x - r, y - r, x + r, y + r,
        #                        start=start_angle, extent=arc_extent,
        #                        style="arc", outline="red", width=8)

        # Needle
        # needle_angle = start_angle + percent * sweep_angle
        needle_angle = start_angle - percent * sweep_angle
        needle_rad = math.radians(needle_angle)
        needle_len = r + 10
        x_end = x + needle_len * math.cos(needle_rad)
        y_end = y - needle_len * math.sin(needle_rad)
        self.canvas.create_line(x, y, x_end, y_end, fill="red", width=3)

        # Dial Center and Texts
        self.canvas.create_oval(x - 90, y - 90, x + 90, y + 90, fill="black", outline="gray")
        self.canvas.create_text(x, y + 10, text=value, fill="white", font=("Helvetica", 32, "bold"))
        self.canvas.create_text(x, y - 40, text=unit, fill="white", font=("Helvetica", 10))

        if label:
            text_id  = self.canvas.create_text(x, y - 65, text=label, fill="grey", font=("Helvetica", 8, "bold"))
            bbox = self.canvas.bbox(text_id)
            # Add padding
            pad = 2
            x1, y1, x2, y2 = bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad

            # Draw rectangle around text
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="white", width=2)
            self.canvas.lift(text_id)
        if gear:
            self.canvas.create_text(x, y - 20, text=gear, fill="#00FFAA", font=("Helvetica", 32, "bold"))


if __name__ == "__main__":
    app = ADASSimulatorGUI()
    app.mainloop()
