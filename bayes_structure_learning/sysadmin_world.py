import numpy as np

from helpers.utils import flip_connection

"""
A class that defines a sysadmin world, with computers connected to each other
A on computer has some chance of turning off. This chance increases the more off computers are connected to it
"""


class SysAdminWorld(object):
    def __init__(self, adj_matrix: np.array):
        # The connectivity of the world is defined by an adjacency matrix of nxn
        self.adj_matrix = adj_matrix
        self.n = adj_matrix.shape[0]

        # State is n bits saying if computer is on or off
        self.state = np.ones((self.n, ), dtype=bool)

        # Probability that a running computer dies by itself and:
        # Probability that any nearby computer kills the nearby computer
        # The prob a computer remains running is (1 - t_prob) * (1-dead_t_prob)^N
        # These probs are normally 1/30 and 1/10 from the paper, but I am doubling them so we get more action
        self.transition_prob = 1.0 / 15.0
        self.nearby_dead_computer_transition_prob = 1.0 / 5.0

    def step(self, a: int):
        """Steps the environment. a is an action, 0 - n-1 reboots a computer, n is do nothing"""

        curr_state = self.state.copy()  # Save the previous state so modifying state in-place doesn't mess anything up

        # Reward = +1 for every alive computer. -1 for rebooting a computer
        reward = sum(self.state)

        # Go through all computers, find how many neighbors they have, and see if they die or not
        for computer_idx in range(self.n):
            parents = self.get_parents(computer_idx)
            N_dead = sum(1 for parent in parents if not curr_state[parent])

            alive_chance = (1 - self.transition_prob) * (1 - self.nearby_dead_computer_transition_prob) ** N_dead
            # print(computer_idx, N_dead, alive_chance)
            if not self.random_sample_less_than(alive_chance):
                self.state[computer_idx] = False

        # Afterwards, if we rebooted a computer, set it to alive, subtract one from reward
        if a < self.n:
            self.state[a] = True
            reward -= 1

        return reward

    def get_parents(self, idx: int):
        # Returns the neighbor (parent?) indices of this computer using the adjaceny matrix
        return [j for j in range(self.n) if self.adj_matrix[j, idx] and j != idx]

    def random_sample_less_than(self, value:float):
        return np.random.uniform() < value

    def reset(self):
        # Turn all computers back on
        self.state = np.ones((self.n, ), dtype=bool)

    def get_factored_state(self):
        return self.state

    def get_flat_state(self):
        # convert state to number (TODO, use binary)
        return None


if __name__ == "__main__":
    # Create a world with three computers and connect comp 0 to 1
    n = 3
    adj_matrix = np.eye(n)
    flip_connection(adj_matrix, 0, 1)

    world = SysAdminWorld(adj_matrix)

    # Run through a few timesteps to make sure the world works properly
    print(world.state)
    for i in range(30):
        action = np.random.randint(0, n+1)
        reward = world.step(action)
        print(action, world.state, reward)
