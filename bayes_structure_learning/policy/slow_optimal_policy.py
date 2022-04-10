import numpy as np


class SlowOptimalPolicy(object):
    # Optimal for the 3 computer world where 0 is connected to 1. Always restart 0 or 1 first.
    # But only with a percentage chance, because I want to gather data of what happens when computers
    # that are connected to each other die
    def __init__(self, n: int):
        self.n = n
        self.action_chance = 0.2

    def select_action(self, state) -> int:
        # If there is a dead computer, restart it, otherwise, do nothing
        # Restarts first computer in the list. But only every fifth of the time
        if np.random.uniform() < self.action_chance:
            if not state[0]:
                return 0
            elif not state[1]:
                return 1
            elif not state[2]:
                return 2
            else:
                return 3
        else:
            return 3
