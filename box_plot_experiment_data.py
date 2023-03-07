"""
Created on 10/21/22 by Ethan Frank

Plots and compares data recorded from experiments. Cumulative reward, time to completion, etc.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

HOME_FOLDER = "/home/edf42001/Documents/College/Thesis/masters-thesis"
TRAIN_FOLDER = "training"

experiment_type = "object_transfer"
experiment_names = ["prison_2023_03_02_10_41_39", "prison_2023_03_02_10_41_48", "prison_2023_03_02_10_42_00",
                    "prison_2023_03_02_10_42_12", "prison_2023_03_02_10_42_17", "prison_2023_03_02_10_42_27", "prison_2023_03_02_10_42_33"]


def load_data(experiment_name: str):
    episode_lengths = []

    path = f"{HOME_FOLDER}/{TRAIN_FOLDER}/{experiment_type}/{experiment_name}"

    if not os.path.exists(path):
        exit(f"Path {path} not found")

    for filename in glob.glob(f"{path}/exp_*"):
        with open(filename, "rt") as f:
            try:
                reward = float(f.readline()[:-1])  # Reward on first line, remove \n
                length = float(f.readline()[:-1])  # Read episode length
                elapsed_time = float(f.readline()[:-1])  # Read time
                episode_lengths.append(length)
            except Exception as e:
                print(f"Error when parsing {filename}, {e}")

    return episode_lengths


def main():
    # Load all the data
    data = []

    for name in experiment_names:
        episode_lengths = load_data(name)
        data.append(episode_lengths)
        print(f"Read {len(episode_lengths)} experiments successfully from {name}")
        print(f"Mean length: {np.mean(episode_lengths)}")

    # Box plot the data
    plt.boxplot(data)
    plt.show()


if __name__ == "__main__":
    main()
