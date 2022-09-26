"""
Created on 9/23/22 by Ethan Frank

For testing actions, rules, win conditions, etc for the taxi/heist combo prison world.
"""

import random
import numpy as np

from environment.prison_world import Prison
from symbolic_stochastic_domains.symbolic_classes import Outcome


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)

    env = Prison(stochastic=False, shuffle_object_names=False)

    for i in range(1000):
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        literals, observation, name_id_map = env.step(action)

        print(action)
        print(literals)
        print(name_id_map)
        print(observation)
        print()

        outcome = Outcome(observation)

        env.visualize(delay=1)
