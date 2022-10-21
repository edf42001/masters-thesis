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
experiment_name = "heist_2022_10_21_11_17_44"
experiment_names_to_compare = ["heist_2022_10_21_12_31_45"]


def load_data(experiment_name: str):
    episode_lengths = []

    path = f"{HOME_FOLDER}/{TRAIN_FOLDER}/{experiment_type}/{experiment_name}"
    for filename in glob.glob(f"{path}/*"):
        with open(filename, "rt") as f:
            try:
                f.readline()  # Ignore reward on first line
                length = float(f.readline()[:-1])  # Read episode length, remove \n
                episode_lengths.append(length)
            except Exception as e:
                print(f"Error when parsing {filename}, {e}")

    return episode_lengths


if __name__ == "__main__":
    # Load all the data
    episode_lengths = load_data(experiment_name)

    print(f"Read {len(episode_lengths)} experiments successfully")

    n = max(100, int((1 + (max(episode_lengths) // 100)) * 100))
    plt.hist(episode_lengths, range=[0, n], bins=n)

    for name in experiment_names_to_compare:
        episode_lengths = load_data(name)
        plt.hist(episode_lengths, range=[0, n], bins=n)

    plt.xlabel("Single Episode Duration")
    plt.ylabel("Frequency")
    plt.title(f"{experiment_type}/{experiment_name} results")
    plt.legend([experiment_name] + experiment_names_to_compare)
    plt.show()
