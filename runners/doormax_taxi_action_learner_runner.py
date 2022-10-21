import random
import logging
import sys
import pickle

from common.data_recorder import DataRecorder
from runners.runner import Runner
from algorithm.action_learning.action_learning_model import ActionLearningModel
from algorithm.action_learning.action_learner import ActionLearner
from environment.taxi_world import TaxiWorld
from policy.action_info_gain import ActionInfoGain


class DoormaxTaxiActionLearnerRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi_doormax_actions'
        self.pkl_name = 'taxi_doormax_actions'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 25
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing
        random.seed(1)

        # Hyperparameters
        params = {
            'discount_factor': 0.987  # Cannot be 1 for Rmax (why?)
        }

        # Learning
        self.env = TaxiWorld(stochastic=self.stochastic, shuffle_actions=True)

        # Assume transition model is already known
        with open("taxi-doormax-model.pkl", 'rb') as f:
            self.doormax_model = pickle.load(f)

        self.model = ActionLearningModel(self.env, self.doormax_model)
        self.planner = ActionInfoGain(self.env.get_num_actions(), self.model)
        self.learner = ActionLearner(self.env, self.model, self.planner, visualize=self.visualize)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = DoormaxTaxiActionLearnerRunner(0)
    runner.run_experiment()
