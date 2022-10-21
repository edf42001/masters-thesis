"""
Created on 10/21/22 by Ethan Frank

Data Recoder object to encapsulate data saving functionality
"""

import os
import time

from runners.runner import Runner

HOME_FOLDER = "/home/edf42001/Documents/College/Thesis/masters-thesis"
TRAIN_FOLDER = "training"


class DataRecorder:
    def __init__(self, runner: Runner, start_time):
        self.runner = runner
        self.exp_time_str = start_time.strftime("%Y_%m_%d_%H_%M_%S")

        self.train_rewards = []
        self.train_steps = []
        self.train_times = []

        self.episode_start_time = 0

    def update(self, steps: int, reward: float, elapsed_time: float):
        self.train_steps.append(steps)
        self.train_rewards.append(reward)
        self.train_times.append(elapsed_time)

    def record_start_time(self):
        self.episode_start_time = time.perf_counter()

    def save_training(self):
        exp_folder = f'{HOME_FOLDER}/{TRAIN_FOLDER}/{self.runner.exp_name}/{self.runner.name}_{self.exp_time_str}'
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder, exist_ok=True)

        with open(f'{exp_folder}/exp_{self.runner.exp_num:03d}.txt', 'wt') as f:
            f.write(",".join([f"{f:.2f}" for f in self.train_rewards]) + "\n")
            f.write(",".join([f"{d:d}" for d in self.train_steps]) + "\n")
            f.write(",".join([f"{f:.2f}" for f in self.train_times]) + "\n")
