from typing import List

import numpy as np


class Hierarchy:

    A_ROOT = None

    env = None
    children = {}
    state_vars = {}
    conditions = {}

    def __init__(self):
        self.actions = list(range(self.A_ROOT + 1))

        # Map each child action to sequential index
        self.children_index_map = {}
        for subtask, children in self.children.items():
            self.children_index_map[subtask] = {child: i for i, child in enumerate(children)}

        # Number of possible assignments to each state variable per subtask
        self.state_arities = {}
        for action, s_vars in self.state_vars.items():
            self.state_arities[action] = [self.env.S_ARITIES[s_var] for s_var in s_vars]

    def get_factored_state(self, state: int) -> List[int]:
        return self.env.get_factored_state(state)

    def is_terminated(self, action: int, state: int, done: bool = False) -> bool:
        """Check if action is terminated is given state"""
        raise NotImplementedError()

    def convert_state(self, state: int, action: int) -> int:
        """Convert state index to abstracted state index"""
        if action == self.A_ROOT:
            return state

        factored_state = self.env.get_factored_state(state)
        abstracted_state = [factored_state[var] for var in self.state_vars[action]]

        return np.ravel_multi_index(abstracted_state, self.state_arities[action])

    def convert_condition(self, condition: List[bool], action: int) -> List[bool]:
        """Create abstract condition using relevant state vars"""
        action_conditions = self.conditions[action]
        return [cond if i in action_conditions else '*' for i, cond in enumerate(condition)]

    def is_primitive(self, action: int) -> bool:
        return not self.children[action]
