import numpy as np
import os
from datetime import datetime

from helpers.utils import flip_connection

from policy.optimal_policy import OptimalPolicy
from policy.slow_optimal_policy import SlowOptimalPolicy


from sysadmin_world import SysAdminWorld


class OptimalRunner(object):
    # Runs experiments with the optimal policy (todo: pass policy as arg?)
    def __init__(self):
        self.policy = SlowOptimalPolicy(3)

        # Create a world with three computers and connect comp 0 to 1
        n = 3
        adj_matrix = np.eye(n)
        flip_connection(adj_matrix, 0, 1)
        self.world = SysAdminWorld(adj_matrix)

        # Store results of experiments
        self.results = None

    def run_experiment(self):
        n_experiments = 10  # Number of times to repeat experiment
        n_iterations = 100

        results = np.empty((n_experiments, n_iterations))

        for i in range(n_experiments):
            self.world.reset()
            rewards = self.run_single_episode(n_iterations)

            results[i, :] = rewards

        self.results = results

        print(np.mean(self.results))

    def run_single_episode(self, n_iterations):
        rewards = np.empty((n_iterations, ))  # Store rewards

        for i in range(n_iterations):
            state = self.world.get_factored_state()
            print(state)

            action = self.policy.select_action(state)

            reward = self.world.step(action)

            rewards[i] = reward

        return rewards

    def save_results(self):
        if not os.path.exists("results"):
            os.mkdir("results")

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        np.save("results/{}.npy".format(now), self.results)


if __name__ == "__main__":
    runner = OptimalRunner()
    runner.run_experiment()
    # runner.save_results()
