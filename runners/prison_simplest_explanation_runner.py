"""
Created on 2/6/23 by Ethan Frank

Simplest explanation runner on prison world. Currently, I give it the ruleset from heist world
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
        self.visualize = True

        self.env = Prison(False, shuffle_object_names=True)

        # Load heist rules to see if it can discover the new objects
        with open("data/heist_learned_data.pkl", 'rb') as f:
            rules, examples, experience_helper = pickle.load(f)

        # Copy so hashes are updated (python gets a new hash seed every run)
        heist_rules = rules.copy()
        heist_examples = examples.copy()
        heist_experiences = experience_helper.copy()

        print(self.env.object_name_map)

        self.model = SimplestExplanationModel(self.env, heist_rules, heist_examples, heist_experiences)
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
        import cProfile
        import pstats

        # profiler = cProfile.Profile()
        # profiler.enable()

        runner = PrisonSimplestExplanationRunner(i, start_time=experiments_start_time)
        runner.run_experiment(save_training=False)

        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats("simplest_explanation_runner_1.prof")
