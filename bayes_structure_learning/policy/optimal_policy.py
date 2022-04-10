class OptimalPolicy(object):
    # Optimal for the 3 computer world where 0 is connected to 1. Always restart 0 or 1 first
    def __init__(self, n: int):
        self.n = n

    def select_action(self, state) -> int:
        # If there is a dead computer, restart it, otherwise, do nothing
        # Restarts first computer in the list
        if not state[0]:
            return 0
        elif not state[1]:
            return 1
        elif not state[2]:
            return 2
        else:
            return 3
