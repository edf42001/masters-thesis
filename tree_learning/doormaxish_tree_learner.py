import numpy as np
from enum import Enum

from tree_learning.decision_node import DecisionNode


# Enum for actions
class ACTION(Enum):
    LEFT = 0
    RIGHT = 1


ATTRIBUTES = ["taxi.x", "door.open"]


class DoormaxishTreeLearner(object):
    """Uses trees directly to learn conditions and effects"""

    def __init__(self, env):
        self.env = env

        # List of effect predictions for each attribute and action
        self.predictions = dict()

        self.init_data_structures()

    def init_data_structures(self):
        for action in list(ACTION):

            # Create a dict for each action
            self.predictions[action] = dict()

            for attr in ATTRIBUTES:

                # Each predictor is a decision tree
                self.predictions[action][attr] = DecisionNode()

    def update_trees(self, data):
        # First, process the data
        for ac, action in enumerate(list(ACTION)):
            for at, attr in enumerate(ATTRIBUTES):
                # Run the classification for each action / attribute pair
                specific_data = self.extract_relavant_data(data, ac, at)

                # Need to have data in order to do predictions
                if len(specific_data) > 0:
                    self.predictions[action][attr].recursively_split(specific_data)

    def extract_relavant_data(self, data, action, attr) -> np.ndarray:
        # Extract all the actions
        data = data[data[:, 10] == action]

        # Extract the conditions
        conditions = data[:, 1:10]

        # Extract the dx for each step based on next and prev state
        d_attr = np.empty((len(data), 1), dtype=int)
        for i, step in enumerate(data):
            state = self.env.get_factored_state(step[0])
            next_state = self.env.get_factored_state(step[11])

            d_attr[i, 0] = next_state[attr] - state[attr]  # Change in effect values (todo: replace with effects?)

        return np.hstack((conditions, d_attr))
