import random
import numpy as np
import logging
import sys
import pickle

from common.plotting.plot import Plot
from runners.runner import Runner
from algorithm.symbolic_domains.object_transfer_model import ObjectTransferModel
from algorithm.symbolic_domains.object_transfer_learner import ObjectTransferLearner
from environment.symbolic_taxi import SymbolicTaxi
from policy.object_transfer_policy import ObjectTransferPolicy


class TaxiObjectTransferRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi'
        self.pkl_name = 'taxi'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 50
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing (I think I have to set both or the keys spawn in different areas)
        random.seed(3)
        np.random.seed(3)

        self.env = SymbolicTaxi(stochastic=self.stochastic, shuffle_object_names=True)

        # Load previously learned model with different object names
        with open("symbolic_taxi_rules_updated.pkl", 'rb') as f:
            symbolic_taxi_ruleset = pickle.load(f)

        self.model = ObjectTransferModel(self.env, symbolic_taxi_ruleset)
        self.planner = ObjectTransferPolicy(self.env.get_num_actions(), self.model)
        self.learner = ObjectTransferLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=100)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = TaxiObjectTransferRunner(0)
    runner.run_experiment()
