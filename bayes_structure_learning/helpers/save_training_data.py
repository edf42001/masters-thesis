import numpy as np
import os
from datetime import datetime

from helpers.utils import flip_connection

from policy.optimal_policy import OptimalPolicy
from policy.slow_optimal_policy import SlowOptimalPolicy


from sysadmin_world import SysAdminWorld

if __name__ == "__main__":
    n_iterations = 10000
    policy = SlowOptimalPolicy(3)

    # Create a world with three computers and connect comp 0 to 1
    n = 3
    adj_matrix = np.eye(n)
    flip_connection(adj_matrix, 0, 1)
    world = SysAdminWorld(adj_matrix)
    # Runs the environment for a bit to collect info about transition function for future testing
    rewards = np.empty((n_iterations, ))  # Store rewards
    states_actions = np.empty((n_iterations, 4))

    for i in range(n_iterations):
        state = world.get_factored_state()
        # print(state)

        action = policy.select_action(state)

        states_actions[i, :3] = state
        states_actions[i, 3] = action

        reward = world.step(action)

        rewards[i] = reward

    np.save("../data/training_data.npy", states_actions)


