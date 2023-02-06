"""
Created on 2/5/23 by Ethan Frank

Runs simplest explanation rule object transfer learning on Heist
"""

import random
import numpy as np
import logging
import sys
import pickle
from datetime import datetime

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.simplest_explanation_model import SimplestExplanationModel
from algorithm.symbolic_domains.simplest_explanation_learner import SimplestExplanationLearner
from environment.symbolic_heist import SymbolicHeist
from policy.simplest_explanation_policy import SimplestExplanationPolicy


class HeistSimplestExplanationRunner(Runner):
    def __init__(self, exp_num, start_time):
        super().__init__()

        self.name = 'heist'
        self.exp_name = 'object_transfer'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 100
        self.num_episodes = 1
        self.visualize = True

        self.env = SymbolicHeist(False, shuffle_object_names=True)

        with open("data/heist_rules.pkl", 'rb') as f:
            heist_rules = pickle.load(f)

        with open("data/heist_examples.pkl", 'rb') as f:
            heist_examples = pickle.load(f)

        # Copy so hashes are updated (python gets a new hash seed every run)
        heist_rules = heist_rules.copy()
        heist_examples = heist_examples.copy()

        print(self.env.object_name_map)

        self.model = SimplestExplanationModel(self.env, heist_rules, heist_examples)
        self.planner = SimplestExplanationPolicy(self.env.get_num_actions(), self.model)
        self.learner = SimplestExplanationLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.data_recorder = DataRecorder(self, start_time)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    random.seed(1)
    np.random.seed(1)

    num_experiments = 1

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder

    for i in range(num_experiments):
        runner = HeistSimplestExplanationRunner(i, start_time=experiments_start_time)
        runner.run_experiment(save_training=False)
