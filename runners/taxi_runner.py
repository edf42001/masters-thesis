import random
import numpy as np
import logging
import sys
from datetime import datetime
from multiprocessing import Pool

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.symbolic_model import SymbolicModel
from algorithm.symbolic_domains.symbolic_learner import SymbolicLearner
from environment.symbolic_taxi import SymbolicTaxi
from policy.symbolic_policy import SymbolicPolicy


class TaxiRunner(Runner):
    def __init__(self, exp_num, start_time):
        super().__init__()

        self.name = 'taxi'
        self.exp_name = 'symbolic_learning'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 200
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = False

        self.env = SymbolicTaxi(stochastic=self.stochastic)

        self.model = SymbolicModel(self.env)
        self.planner = SymbolicPolicy(self.env.get_num_actions(), self.model)
        self.learner = SymbolicLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.data_recorder = DataRecorder(self, start_time)


def run_single_experiment(experiment_num: int):
    # I don't know how to pass additional args, so experiments_start_time will just be read
    # from the below scope.

    # Also, reset the random seed, otherwise, they all have the same seed
    np.random.seed(None)
    random.seed()
    runner = TaxiRunner(experiment_num, start_time=experiments_start_time)
    runner.run_experiment(save_training=True)

    import pickle
    with open("data/symbolic_taxi_rules.pkl", 'wb') as f:
        pickle.dump(runner.model.ruleset, f)

    with open("data/taxi_examples.pkl", 'wb') as f:
        pickle.dump(runner.model.examples, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Using a random seed makes all the processes have the same seed

    num_experiments = 1

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder

    experiment_numbers = np.arange(num_experiments, dtype=int)

    with Pool(processes=6) as pool:
        results = pool.imap_unordered(run_single_experiment, experiment_numbers, chunksize=5)

        for _ in results:
            pass
