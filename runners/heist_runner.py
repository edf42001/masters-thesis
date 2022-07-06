import random
import numpy as np
import logging
import sys

from common.plotting.plot import Plot
from runners.runner import Runner
from algorithm.symbolic_domains.symbolic_model import SymbolicModel
from algorithm.symbolic_domains.symbolic_learner import SymbolicLearner
from environment.symbolic_heist import SymbolicHeist
from policy.symbolic_policy import SymbolicPolicy


class HeistRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'heist'
        self.pkl_name = 'heist'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 200
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing (I think I have to set both or the keys spawn in different areas)
        random.seed(1)
        np.random.seed(1)

        self.env = SymbolicHeist(stochastic=self.stochastic)

        self.model = SymbolicModel(self.env)
        self.planner = SymbolicPolicy(self.env.get_num_actions(), self.model)
        self.learner = SymbolicLearner(self.env, self.model, self.planner, visualize=self.visualize, delay=10)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = HeistRunner(0)
    runner.run_experiment()
