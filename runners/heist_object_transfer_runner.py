import random
import numpy as np
import logging
import sys
import pickle

from common.plotting.plot import Plot
from runners.runner import Runner
from algorithm.symbolic_domains.object_transfer_model import ObjectTransferModel
from algorithm.symbolic_domains.object_transfer_learner import ObjectTransferLearner
from environment.symbolic_heist import SymbolicHeist
from policy.object_transfer_policy import ObjectTransferPolicy


class HeistObjectTransferRunner(Runner):
    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi'
        self.pkl_name = 'taxi'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 100
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing (I think I have to set both or the keys spawn in different areas)
        random.seed(6)
        np.random.seed(6)

        self.env = SymbolicHeist(stochastic=self.stochastic, shuffle_object_names=True)

        # Load previously learned model with different object names
        with open("symbolic_heist_rules.pkl", 'rb') as f:
            symbolic_heist_rules = pickle.load(f)

        print(self.env.object_name_map)

        self.model = ObjectTransferModel(self.env, symbolic_heist_rules)
        self.planner = ObjectTransferPolicy(self.env.get_num_actions(), self.model)
        self.learner = ObjectTransferLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = HeistObjectTransferRunner(0)
    runner.run_experiment()
