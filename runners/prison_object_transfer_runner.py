"""
Created on 10/4/22 by Ethan Frank

Object transfer runner for the combined heist/taxi world prison world.
"""

import random
import numpy as np
import logging
import sys
import pickle
from datetime import datetime

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.object_transfer_model import ObjectTransferModel
from algorithm.symbolic_domains.object_transfer_learner import ObjectTransferLearner
from environment.prison_world import Prison
from policy.object_transfer_policy import ObjectTransferPolicy


class PrisonObjectTransferRunner(Runner):
    def __init__(self, exp_num, start_time):
        super().__init__()

        self.name = 'prison'
        self.exp_name = 'object_transfer'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 100
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        self.env = Prison(stochastic=self.stochastic, shuffle_object_names=True)

        # Load previously learned model with different object names
        with open("data/symbolic_prison_rules.pkl", 'rb') as f:
            symbolic_taxi_ruleset = pickle.load(f)

        self.model = ObjectTransferModel(self.env, symbolic_taxi_ruleset)
        self.planner = ObjectTransferPolicy(self.env.get_num_actions(), self.model)
        self.learner = ObjectTransferLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.data_recorder = DataRecorder(self, start_time)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    random.seed(1)
    np.random.seed(1)

    num_experiments = 50

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder

    for i in range(num_experiments):
        runner = PrisonObjectTransferRunner(i, start_time=experiments_start_time)
        runner.run_experiment(save_training=True)
