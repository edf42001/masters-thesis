"""
Created on 10/21/22 by Ethan Frank

Plots and compares data recorded from experiments. Cumulative reward, time to completion, etc.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np

HOME_FOLDER = "/home/edf42001/Documents/College/Thesis/masters-thesis"
TRAIN_FOLDER = "training"

def load_data(experiment_type: str, experiment_name: str):
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

    return np.array(episode_lengths)


def make_plot(experiment_type, experiment_names, title, max_steps=-1):
    # Load example data to get length
    episode_lengths = load_data(experiment_type, experiment_names[0])
    n = max(100, int((1 + (max(episode_lengths) // 100)) * 100))

    # Clear previously called plot since we aren't using plot.show
    plt.figure()

    for name in experiment_names:
        episode_lengths = load_data(experiment_type, name)

        print(f"Read {len(episode_lengths)} experiments successfully")

        if max_steps > 0:
            start_size = len(episode_lengths)
            episode_lengths = episode_lengths[episode_lengths != max_steps]
            end_size = len(episode_lengths)
            failures = start_size - end_size
            print(f"Removed {failures} failures ({failures/start_size:.3f}%)")

        plt.hist(episode_lengths, range=[0, n], bins=int(n/4), density=True)

        print(f"Mean: {np.mean(episode_lengths):.2f}, Std: {np.std(episode_lengths):.2f}")
    print()

    plt.xlabel("Single Episode Duration")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(title + "_symbolic_learning.png", pad_inches=0.3)
    # plt.show()


if __name__ == "__main__":
    experiment_type = "symbolic_learning"

    make_plot(experiment_type, ["taxi_runner_300_trials"], "Taxi (all terms)", max_steps=200)
    # make_plot(experiment_type, ["heist_runner_300_trials"], "Heist (all relations)", max_steps=250)
    # make_plot(experiment_type, ["prison_runner_300_trials"], "Prison (all relations)", max_steps=300)

    make_plot(experiment_type, ["taxI_300_only_necessary_relations"], "Taxi (reduced terms)", max_steps=200)
    # make_plot(experiment_type, ["heist_300_only_necessary_relations"], "Heist (reduced relations)", max_steps=250)
    # make_plot(experiment_type, ["prison_300_only_necessary_relations"], "Prison (reduced relations)", max_steps=300)
