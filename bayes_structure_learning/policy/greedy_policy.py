import numpy as np


class GreedyPolicy(object):
    # Always restarts the first computer that has died
    def __init__(self, n: int):
        self.n = n

    def select_action(self, state: np.array) -> int:
        # If there is a dead computer, restart it, otherwise, do nothing
        # Restarts first computer in the list
        if False in state:
            return state.argmin()
        else:
            return self.n
