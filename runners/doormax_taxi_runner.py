import random
import logging
import sys
import pickle

from common.plotting.plot import Plot
from runners.runner import Runner
from policy.value_iteration import ValueIteration
from algorithm.doormax.doormax_ruleset import DoormaxRuleset
from algorithm.doormax.doormax_simulator import DoormaxSimulator
from environment.taxi_world import TaxiWorld


class DoormaxTaxiRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi_doormax'
        self.pkl_name = 'taxi_doormax'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 10
        self.num_episodes = 1
        self.eval_episodes = 1
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

        self.model = DoormaxRuleset(self.env)

        # LOAD, for testing
        # with open(f'model_{self.name}_{self.pkl_name}_{self.exp_num}.pkl', 'rb') as f:
        #     self.model = pickle.load(f)

        self.planner = ValueIteration(self.env.get_num_states(), self.env.get_num_actions(),
                                      params['discount_factor'], self.env.get_rmax(), self.model)
        self.learner = DoormaxSimulator(self.env, self.model, self.planner, visualize=self.visualize)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)

        # IDEAS
        # Replace Q iteration with own custom value iteration
        # Unit tests of some sort
        # Switch to one-d taxi world
        # Contradictory effects are how the taxi gets stuck in the top left corner because it can't
        # tell if up sets y to 0 or subtracts one from it but these are the same thing, so it doesn't matter
        # Need to test more methodically. How? Should I change taxi's state variables to my state variables?
        # What about end of episode? I think the taxi thinks it can keep going after pass is set to num_locations + 1
        # Reward should be based on action, not on state
        # Reenable nochange effects. Compare exactly to other taxi to make sure we have a one to one match.
        # Do this by setting state deterministicley and inputting actions 1 by 1. <-- good idea


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN, stream=sys.stdout)

    runner = DoormaxTaxiRunner(0)
    runner.run_experiment()

    # with open(f'model_{runner.name}_{runner.pkl_name}_{runner.exp_num}.pkl', 'wb') as f:
    #     pickle.dump(runner.model, f)
