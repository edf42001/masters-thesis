from algorithm.distance_metric_option_discovery_algorithm import DistanceMetricOptionDiscoveryAlgorithm
from common.plotting.plot import Plot
from runners.runner import Runner
from policy.distance_metric_option_discovery_policy import DistanceMetricOptionDiscoverPolicy
from policy.samplers.epsilon_greedy import EpsilonGreedy
from environment.two_room_env import TwoRoomEnv


class DistanceMetricRunner(Runner):
    def __init__(self, experiment_num):
        super().__init__()

        self.name = 'distance_metric_options'
        self.pkl_name = 'distance_metric_options'
        self.exp_num = experiment_num

        # Experiment parameters
        self.max_steps = 4000
        self.num_episodes = 100
        self.eval_episodes = 20
        self.eval_timer = 10
        self.stochastic = False
        self.all_goals = False
        self.visualize = False

        # Hyperparameters
        params = {
            'discount_factor': 0.9,  # Q-Learning
            'learning_rate': 0.1,  # Q-Learning
            'a_decay': 1,  # Q-Learning
            'min_a': 0.1,  # Q-Learning
            'epsilon': 1,  # E-Greedy Exploration
            'e_decay': 1,  # E-Greedy Exploration
            'min_e': 0.01,  # E-Greedy Exploration
        }

        # Learning
        self.env = TwoRoomEnv()
        self.policy = DistanceMetricOptionDiscoverPolicy(self.env.get_num_states(), self.env.get_num_actions(), params)
        self.sampler = EpsilonGreedy(params)
        self.learner = DistanceMetricOptionDiscoveryAlgorithm(self.env, self.policy, self.sampler, visualize=self.visualize)
        self.plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    experiment_num = 0
    controller = DistanceMetricRunner(experiment_num)
    controller.run_experiment()
