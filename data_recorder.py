##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## data_recorder
##

import csv
import os

DATAS_DIR = "Datas"
DECIMAL_PLACES = 5

os.makedirs(DATAS_DIR, exist_ok=True)

def get_unique_filepath(base_name: str):
    """Generate a unique filename by appending a number if needed."""
    i = 1
    filepath = os.path.join(DATAS_DIR, f"{base_name}_{i}.csv").replace("\\", "/")
    while os.path.exists(filepath):
        i += 1
        filepath = os.path.join(DATAS_DIR, f"{base_name}_{i}.csv").replace("\\", "/")
    return filepath

class DataRecorder:
    """Handles data recording (only in manual mode)."""
    def __init__(self, nb_rays: int, fov: int):
        self.num_rays = nb_rays
        self.fov = fov
        self.filepath = get_unique_filepath("data")
        self.csv_file = open(self.filepath, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=',')

        if self.csv_file.tell() == 0:
            self.csv_file.write(f"# Number of rays: {self.num_rays} | FOV: {self.fov}\n")
            header = ["input_speed", "input_steering", "speed", "steering"] \
                    + ["delta_speed", "delta_steering", "angle_closest_ray", "avg_ray", "std_ray", "min_ray", "max_ray", \
                      "avg_ray_left", "avg_ray_center", "avg_ray_right", "ray_balance", "acceleration"] \
                    + [f"ray_{i}" for i in range(1, self.num_rays + 1)]

            self.csv_writer.writerow(header)

    def write_data(self, data: dict, input_speed: float, input_steering: float):
        """Save collected data along with the input commands"""

        input_speed = round(input_speed, DECIMAL_PLACES)
        input_steering = round(input_steering, DECIMAL_PLACES)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, DECIMAL_PLACES)

        row = [input_speed, input_steering, data["speed"], data["steering"]]
        row += [data["delta_speed"], data["delta_steering"], data["angle_closest_ray"], data["avg_ray"], data["std_ray"], data["min_ray"],
                data["max_ray"], data["avg_ray_left"], data["avg_ray_center"],
                data["avg_ray_right"], data["ray_balance"], data["acceleration"]]
        row += [data[f"ray_{i}"] for i in range(1, self.num_rays + 1)]

        self.csv_writer.writerow(row)

    def __del__(self):
        self.csv_file.close()
