
import numpy as np
import random

from environment.hierarchy.hierarchy import Hierarchy
from policy.policy import Policy


class HierarchicalPolicy(Policy):
    def __init__(self, hierarchy: Hierarchy, params):
        self.hierarchy = hierarchy
        self.discount_factor = params['discount_factor']
        self.learning_rate = params['learning_rate']
        self.b_learning_rate = params.get('b_learning_rate', self.learning_rate)
        self.decay = params['a_decay']
        self.min_alpha = params['min_a']

        # Set up value and completion function storage TODO
        # self.V, self.C = {}, {}
        # init_value = -1
        # for action in self.the_hier.actions:
        #     num_states = np.prod(self.the_hier.state_arities[action])
        #     if self.the_hier.is_primitive(action):
        #         # Track the value of this primitive in all possible abstracted states
        #         self.V[action] = np.full(num_states, init_value, dtype='float32')
        #     else:
        #         # Track the completion function for this subtask
        #         # Keep one value for each abstracted state and child action
        #         num_children = len(self.the_hier.children[action])
        #         self.C[action] = np.full((num_states, num_children), init_value, dtype='float32')
