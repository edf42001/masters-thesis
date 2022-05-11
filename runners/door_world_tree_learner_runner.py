"""Uses a tree learner to learn conditions and effects and figure out how to get to the goal"""

import logging
import random
import numpy as np

from environment.door_world import DoorWorld
from tree_learning.doormaxish_tree_learner import DoormaxishTreeLearner


def select_action(world) -> int:
    """Uses the world to select the best action"""
    return random.randint(0, 1)


if __name__ == "__main__":
    # Could log to stdout with a streamhandler, but instead, am just going to disable the filename here
    # logging.basicConfig(level=logging.INFO, filename="log.log", filemode='w',
    #                     format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
    logging.basicConfig(level=logging.INFO)

    world = DoorWorld()

    num_steps = 100
    np.random.seed(1)

    # Store complete transition history data
    step_data = np.empty((num_steps, 1 + 9 + 1 + 1), dtype=int)

    tree_learner = DoormaxishTreeLearner(world)
    logging.info(tree_learner.predictions)

    for i in range(num_steps):
        action = select_action(world)

        state = world.get_state()
        conditions = world.get_condition(state)

        observation = world.step(action=action)

        done = world.end_of_episode()
        reward = world.get_last_reward()
        next_state = world.get_state()

        logging.info("Conditions {}".format("".join(["1" if c else "0" for c in conditions])))
        logging.info("Taking action {}. Done {} obs {} reward {}".format(action, done, observation, reward))
        logging.info(world.visualize())

        # Store transition data
        step_data[i, 0] = state
        step_data[i, 1:10] = conditions
        step_data[i, 10] = action
        step_data[i, 11] = next_state

        # Update all the trees based on the data so far
        tree_learner.update_trees(step_data[:i+1])
        logging.info(tree_learner.predictions)

        if done:
            logging.info("Restarting world, i = {}".format(i))
            world.restart()
