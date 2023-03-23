"""
Created on 10/21/22 by Ethan Frank

Plots and compares data recorded from experiments. Cumulative reward, time to completion, etc.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np

HOME_FOLDER = "/home/edf42001/Documents/College/Thesis/masters-thesis"
TRAIN_FOLDER = "training"

experiment_type = "symbolic_learning"
experiment_names = ["taxi_runner_300_trials", "taxI_300_only_necessary_relations"]
title = "Taxi"


def load_data(experiment_name: str):
    episode_lengths = []

    path = f"{HOME_FOLDER}/{TRAIN_FOLDER}/{experiment_type}/{experiment_name}"
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


if __name__ == "__main__":
    # Load example data to get length
    episode_lengths = load_data(experiment_names[0])
    n = max(100, int((1 + (max(episode_lengths) // 100)) * 100))

    for name in experiment_names:
        episode_lengths = load_data(name)
        plt.hist(episode_lengths, range=[0, n], bins=n, density=True)

        print(f"Read {len(episode_lengths)} experiments successfully")
        print(f"Mean: {np.mean(episode_lengths):.2f}, Std: {np.std(episode_lengths):.2f}")

    plt.xlabel("Single Episode Duration")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()
