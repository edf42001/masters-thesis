import random
import logging
import sys
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple

import numpy as np

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.symbolic_model import SymbolicModel
from algorithm.symbolic_domains.symbolic_learner import SymbolicLearner
from environment.prison_world import Prison
from policy.symbolic_policy import SymbolicPolicy


class PrisonRunner(Runner):
    def __init__(self, exp_num, start_time):
        super().__init__()

        self.name = 'prison'
        self.exp_name = 'symbolic_learning'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 250
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = False

        self.env = Prison(stochastic=self.stochastic)

        self.model = SymbolicModel(self.env)
        self.planner = SymbolicPolicy(self.env.get_num_actions(), self.model)
        self.learner = SymbolicLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=1)
        self.data_recorder = DataRecorder(self, start_time)


def run_single_experiment(data: Tuple[int, str]):
    # Also, reset the random seed, otherwise, they all have the same seed
    np.random.seed(None)
    random.seed()
    experiment_num, start_time = data
    runner = PrisonRunner(experiment_num, start_time=start_time)
    runner.run_experiment(save_training=True)


def main():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    num_experiments = 10

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder

    experiment_numbers = np.arange(num_experiments, dtype=int)

    # Only way to pass both data
    data = [(num, experiments_start_time) for num in experiment_numbers]

    with Pool(processes=6) as pool:
        results = pool.imap_unordered(run_single_experiment, data, chunksize=5)

        for _ in results:
            pass


if __name__ == '__main__':
    main()
