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
        self.max_steps = 150
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing (I think I have to set both or the keys spawn in different areas)
        random.seed(1)
        np.random.seed(1)

        self.env = SymbolicTaxi(stochastic=self.stochastic, shuffle_object_names=True)

        # Load previously learned model with different object names
        with open("symbolic_taxi_rules.pkl", 'rb') as f:
            symbolic_taxi_ruleset = pickle.load(f)

        prior_object_names = ["taxi", "pass", "dest", "wall"]
        print(prior_object_names)
        print(self.env.get_object_names())

        # Assign initial probability for each object being each other object
        likelihood_map = np.ones((len(prior_object_names), len(self.env.get_object_names())))
        likelihood_map[0, 1:] = 0  # Set taxi to be known as taxi
        likelihood_map[1:, 0] = 0
        likelihood_map /= (len(self.env.get_object_names()) - 1)
        likelihood_map[0, 0] = 1  # Again, set this to 1, we know taxi is taxi
        print(likelihood_map)

        self.model = ObjectTransferModel(self.env)
        self.planner = ObjectTransferPolicy(self.env.get_num_actions(), self.model)
        self.learner = ObjectTransferLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = TaxiObjectTransferRunner(0)
    # runner.run_experiment()
