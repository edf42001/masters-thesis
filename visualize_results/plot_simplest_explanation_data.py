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

experiment_type = ""

experiment_names = ["simplest_explanation/taxi_2023_03_24_13_57_43", "simplest_explanation/heist_2023_03_24_13_58_14", "simplest_explanation/prison_2023_03_24_14_00_52",
                    "object_transfer/taxi_2023_03_23_16_59_18", "object_transfer/heist_2023_03_23_16_59_29", "object_transfer/prison_2023_03_07_14_05_26"]


# TODO: Why is prison broken?

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
    plt.boxplot(data, labels=["Taxi", "Heist", "Prison"])
    plt.title("Taxi")
    plt.show()


if __name__ == "__main__":
    main()



# reward = [float(value) for value in f.readline()[:-1].split(",")]  # Reward on first line, remove \n
# length = [float(value) for value in f.readline()[:-1].split(",")]  # Read episode length
# elapsed_time = [float(value) for value in f.readline()[:-1].split(",")]  # Read time
# episode_lengths.append(length)
