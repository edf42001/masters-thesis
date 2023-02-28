import random
import numpy as np
import logging
import sys
import pickle
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, List

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.symbolic_domains.object_transfer_model import ObjectTransferModel
from algorithm.symbolic_domains.object_transfer_learner import ObjectTransferLearner
from environment.symbolic_taxi import SymbolicTaxi
from policy.object_transfer_policy import ObjectTransferPolicy


class TaxiObjectTransferRunner(Runner):
    def __init__(self, exp_num, start_time, known_objects):
        super().__init__()

        self.name = 'taxi'
        self.exp_name = 'object_transfer'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 50
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = False

        self.env = SymbolicTaxi(stochastic=self.stochastic, shuffle_object_names=True, known_objects=known_objects)

        # Load previously learned model with different object names
        with open("data/symbolic_taxi_rules.pkl", 'rb') as f:
            symbolic_taxi_ruleset = pickle.load(f)

        self.model = ObjectTransferModel(self.env, symbolic_taxi_ruleset)
        self.planner = ObjectTransferPolicy(self.env.get_num_actions(), self.model)
        self.learner = ObjectTransferLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=100)
        self.data_recorder = DataRecorder(self, start_time)


def run_single_experiment(data: Tuple[int, str, List]):
    # Also, reset the random seed, otherwise, they all have the same seed
    np.random.seed(None)
    random.seed()
    experiment_num, start_time, known_objects = data
    runner = TaxiObjectTransferRunner(experiment_num, start_time=start_time, known_objects=known_objects)
    runner.run_experiment(save_training=True)


def main(**kwargs):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    num_experiments = 100

    experiments_start_time = datetime.now()  # Used for putting all experiments in common folder
    experiment_numbers = np.arange(num_experiments, dtype=int)
    known_objects = kwargs["known_objects"] if "known_objects" in kwargs else None

    data = [(num, experiments_start_time, known_objects) for num in experiment_numbers]  # Only way to pass both exp num and start time

    with Pool(processes=6) as pool:
        results = pool.imap_unordered(run_single_experiment, data, chunksize=5)

        for _ in results:
            pass


if __name__ == '__main__':
    main()
