import logging
import numpy as np

from environment.door_world import DoorWorld
from tree_learning.decision_node import DecisionNode


def split_data(data, condition_idx):
    """Takes in data and splits it into two based on the condition idx"""

    # left is true, right is false
    data_left = data[data[:, condition_idx] == 1]
    data_right = data[data[:, condition_idx] == 0]

    return data_left, data_right


if __name__ == "__main__":
    # State = 0, conditions = 1-9, action = 10, next_state = 11
    # State is (x, door_open)
    data = np.loadtxt("../runners/step_data.csv", delimiter=",", dtype=int)

    world = DoorWorld()
    cond_names = ["touch_L_wall", "touch_R_wall", "touch_L_door", "touch_R_door", "touch_L_goal", "touch_R_goal",
                  "touch_L_switch", "touch_R_switch", "open_door"]

    # Extract all right actions
    data = data[data[:, 10] == 1]
    conditions = data[:, 1:10]

    # Extract the dx for each step based on next and prev state
    dx = np.empty((len(data), 1), dtype=int)
    for i, step in enumerate(data):
        state = world.get_factored_state(step[0])
        next_state = world.get_factored_state(step[11])

        dx[i, 0] = next_state[0] - state[0]  # Change in x values
        # dx[i, 0] = next_state[1] - state[1]  # Change in door_open boolean

    # Now we have a classification problem: conditions as inputs, dx as output
    # Tree is deterministic, will have no error. Use a tree to describe the classification
    data = np.hstack((conditions, dx))
    # print(data)

    # See what trees are generated when we only have a bit of the data
    for i in range(1, len(data)):
        shortened_data = data[:i, :]

        node = DecisionNode()
        node.recursively_split(shortened_data)
        print("i = {}".format(i))
        node.print()
        print()

    # At each leaf of tree, we can have teh agent test it's assumptions by trying every combination
    # i.e., ram it's head into everything
    # Is this exactly the same as doormax?
    print(node)
