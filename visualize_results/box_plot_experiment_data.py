"""
Created on 3/05/23 by Ethan Frank

Plots and compares data recorded from experiments. Cumulative reward, time to completion, etc.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

HOME_FOLDER = "/home/edf42001/Documents/College/Thesis/masters-thesis"
TRAIN_FOLDER = "training"

experiment_type = "object_transfer"
# experiment_names = ["taxi_2023_03_23_16_59_18", "taxi_2023_03_23_16_59_21", "taxi_2023_03_23_16_59_24",
#                     "taxi_2023_03_23_16_59_27", "taxi_2023_03_24_18_03_37"]
# experiment_names = ["heist_2023_03_23_16_59_29", "heist_2023_03_23_16_59_42", "heist_2023_03_23_16_59_55",
#                     "heist_2023_03_23_17_00_13", "heist_2023_03_23_17_00_19", "heist_2023_03_24_18_03_41"]
experiment_names = ["prison_2023_03_23_17_00_34", "prison_2023_03_23_17_01_28",
                    "prison_2023_03_23_17_02_23", "prison_2023_03_23_17_03_53",
                    "prison_2023_03_23_17_04_43", "prison_2023_03_23_17_05_12", "prison_2023_03_24_18_03_52"]

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
        print(f"Mean: {np.mean(episode_lengths):.2f}, Std: {np.std(episode_lengths):.2f}")

    # Box plot the data
    # plt.boxplot(data, labels=["None", "Wall", "Key", "Gem", "Lock", "All"])
    # plt.boxplot(data, labels=["None", "Wall", "Pass", "Dest", "All"])
    plt.boxplot(data, labels=["None", "Wall", "Key", "Lock", "Pass", "Dest", "All"])
    # plt.ylim([51, 82])  # For prison, scaling until I can fix the 300 errors
    plt.title("Prison")
    plt.show()


if __name__ == "__main__":
    main()
