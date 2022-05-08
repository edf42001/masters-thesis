import logging
import numpy as np

from environment.door_world import DoorWorld


def split_data(data, condition_idx):
    """Takes in data and splits it into two based on the condition idx"""

    # left is true, right is false
    data_left = data[conditions[:, condition_idx] == 1]
    data_right = data[conditions[:, condition_idx] == 0]

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

        dx[i, 0] = next_state[0] - state[0]

    # Now we have a classification problem: conditions as inputs, dx as output
    # Tree is deterministic, will have no error. Use a tree to describe the classification
    data = np.hstack((conditions, dx))
    print(data)

    for condition_idx in range(0, 9):
        left, right = split_data(data, condition_idx)

        left_moves = np.count_nonzero(left[:, -1])
        right_moves = np.count_nonzero(right[:, -1])
        print("{}: {}/{}, {}/{}".format(cond_names[condition_idx], left_moves, len(left) - left_moves,
                                        right_moves, len(right) - right_moves))

