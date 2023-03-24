"""
Created on 2/6/23 by Ethan Frank

Simplest explanation runner on prison world. Currently, I give it the ruleset from heist world
"""

import random
from multiprocessing import Pool
from typing import Tuple, List

import numpy as np
import logging
import sys
import pickle
from datetime import datetime

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.simplest_explanation_model import SimplestExplanationModel
from algorithm.symbolic_domains.simplest_explanation_learner import SimplestExplanationLearner
from environment.prison_world import Prison
from policy.simplest_explanation_policy import SimplestExplanationPolicy


class PrisonSimplestExplanationRunner(Runner):
    def __init__(self, exp_num, start_time):
        super().__init__()

        self.name = 'prison'
        self.exp_name = 'simplest_explanation'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 150
        self.num_episodes = 2
        self.visualize = False

        self.env = Prison(False, shuffle_object_names=True)

        # Load heist rules to see if it can discover the new objects
        with open("data/prison_learned_data.pkl", 'rb') as f:
            rules, examples, experience_helper = pickle.load(f)

        # Copy so hashes are updated (python gets a new hash seed every run)
        rules = rules.copy()
        examples = examples.copy()
        experiences = experience_helper.copy()

        print(self.env.object_name_map)

        self.model = SimplestExplanationModel(self.env, rules, examples, experiences)
        self.planner = SimplestExplanationPolicy(self.env.get_num_actions(), self.model)
        self.learner = SimplestExplanationLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.data_recorder = DataRecorder(self, start_time)


def run_single_experiment(data: Tuple[int, str]):
    # Also, reset the random seed, otherwise, they all have the same seed
    np.random.seed(None)
    random.seed()
    experiment_num, start_time = data

    runner = PrisonSimplestExplanationRunner(experiment_num, start_time=start_time)
    runner.run_experiment(save_training=True)


def main():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    num_experiments = 10

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder
    experiment_numbers = np.arange(num_experiments, dtype=int)

    data = [(num, experiments_start_time) for num in experiment_numbers]  # Only way to pass both exp num and start time

    with Pool(processes=6) as pool:
        results = pool.imap_unordered(run_single_experiment, data, chunksize=5)

        for _ in results:
            pass


if __name__ == "__main__":
    main()
