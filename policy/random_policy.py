import random

from policy.policy import Policy


class RandomPolicy(Policy):
    """Chooses an action from the N available randomly"""

    def __init__(self, actions: int):
        self.num_actions = actions

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # Choose a random action
        return random.randint(0, self.num_actions-1)
