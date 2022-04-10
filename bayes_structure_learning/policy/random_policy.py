import numpy as np


class RandomPolicy(object):
    # Randomly restarts any computer or does nothing
    def __init__(self, n: int):
        self.n = n

    def select_action(self, state) -> int:
        return np.random.randint(0, self.n+1)