from algorithm.MaxQ import MaxQ
# from common.plotting import Plot TODO
from runners.runner import Runner
from policy.hierarchical_policy import HierarchicalPolicy
from policy.samplers.epsilon_greedy import EpsilonGreedy
from environment.taxi_world import TaxiWorld


class MaxQTaxiRunner(Runner):

    def __init__(self, exp_num):
        self.name = 'taxi_maxq'
        self.pkl_name = 'df1_lr01_ep02'
        self.exp_num = exp_num

        # Experiment parameters
        self.max_steps = 300
        self.num_episodes = 500
        self.eval_episodes = 20
        self.eval_timer = 10
        self.stochastic = True
        self.all_goals = True
        self.visualize = False

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
        self.policy = HierarchicalPolicy(self.env.hierarchy, params)
        self.sampler = EpsilonGreedy(params)
        self.learner = MaxQ(self.env, self.policy, self.sampler, visualize=self.visualize, all_goals=self.all_goals)
        # self.the_plot = Plot(self, self.eval_episodes, self.eval_timer)


if __name__ == '__main__':
    for exp_num in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        controller = MaxQTaxiRunner(exp_num)
        controller.run_experiment()
