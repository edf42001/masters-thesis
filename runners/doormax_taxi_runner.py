from algorithm.doormax import Doormax
from common.plotting.plot import Plot
from runners.runner import Runner
from policy.hierarchical_policy import HierarchicalPolicy
from policy.samplers.epsilon_greedy import EpsilonGreedy
from environment.taxi_world import TaxiWorld


class DoormaxTaxiRunner(Runner):

    def __init__(self, exp_num):
        super().__init__()

        self.name = 'taxi_doormax'
        self.pkl_name = 'taxi_doormax'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 30
        self.num_episodes = 1
        self.eval_episodes = 20
        self.eval_timer = 10
        self.stochastic = False
        self.all_goals = False
        self.visualize = True

        # Hyperparameters
        params = {
            'discount_factor': 1,  # Q-Learning
            'learning_rate': 0.1,  # Q-Learning
            'a_decay': 1,  # Q-Learning
            'min_a': 0.1,  # Q-Learning
            'epsilon': 0.2,  # E-Greedy Exploration
            'e_decay': 1,  # E-Greedy Exploration
            'min_e': 0.01,  # E-Greedy Exploration
        }

        # Learning
        self.env = TaxiWorld(stochastic=self.stochastic)
        self.learner = Doormax(self.env, visualize=self.visualize, all_goals=self.all_goals)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    runner = DoormaxTaxiRunner(0)
    runner.run_experiment()
