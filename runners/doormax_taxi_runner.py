import random
import logging

from common.plotting.plot import Plot
from runners.runner import Runner
from policy.value_iteration import ValueIteration
from algorithm.doormax.doormax import Doormax
from algorithm.doormax.doormax_simulator import DoormaxSimulator
from environment.taxi_world import TaxiWorld


class DoormaxTaxiRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi_doormax'
        self.pkl_name = 'taxi_doormax'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 100
        self.num_episodes = 1
        self.eval_episodes = 20
        self.eval_timer = 10
        self.stochastic = False
        self.use_outcomes = False
        self.visualize = False

        # For testing
        random.seed(1)

        # Hyperparameters
        params = {
            'discount_factor': 0.95  # Cannot be 1 for Rmax (why?)
        }

        # Learning
        self.env = TaxiWorld(stochastic=self.stochastic, use_outcomes=self.use_outcomes)
        self.model = Doormax(self.env)
        self.planner = ValueIteration(self.env.get_num_states(), self.env.get_num_actions(),
                                      params['discount_factor'], self.env.get_rmax(), self.model)
        self.learner = DoormaxSimulator(self.env, self.model, self.planner, visualize=self.visualize)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    runner = DoormaxTaxiRunner(0)
    runner.run_experiment()
