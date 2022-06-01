import random
import logging
import sys

from common.plotting.plot import Plot
from runners.runner import Runner
from algorithm.symbolic_domains.symbolic_model import SymbolicModel
from algorithm.symbolic_domains.symbolic_learner import SymbolicLearner
from environment.symbolic_door_world import SymbolicDoorWorld
from policy.symbolic_policy import SymbolicPolicy


class DoorWorldSymbolicLearnerRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'door_world_symbolic'
        self.pkl_name = 'door_world_symbolic'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 10
        self.num_episodes = 1
        self.stochastic = False
        self.visualize = True

        # For testing
        random.seed(1)

        # Hyperparameters
        params = {
            'discount_factor': 0.95
        }

        # Learning
        self.env = SymbolicDoorWorld(stochastic=self.stochastic)

        self.model = SymbolicModel(self.env)
        self.planner = SymbolicPolicy(self.env.get_num_actions(), self.model)
        self.learner = SymbolicLearner(self.env, self.model, self.planner, visualize=self.visualize)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    runner = DoorWorldSymbolicLearnerRunner(0)
    runner.run_experiment()
