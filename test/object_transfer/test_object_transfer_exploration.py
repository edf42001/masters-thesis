"""
Created on 8/2/22 by Ethan Frank


Test "experiment design". What should the agent do in order to figure out
which object are which? What actions should it take
"""


import random
import pickle

import numpy as np

from environment.symbolic_taxi import SymbolicTaxi
from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.symbolic_classes import ExampleSet
from symbolic_stochastic_domains.object_transfer import information_gain_of_action, information_gain_of_state

num_actions = 6

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    examples = ExampleSet()

    # These will eventually be stored in a class as a member variable for easy access
    env = SymbolicHeist(stochastic=False, shuffle_object_names=True)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    # Load previously learned model with different object names
    with open("../runners/symbolic_heist_rules.pkl", 'rb') as f:
        previous_ruleset = pickle.load(f)

    print("Object name map:")
    print(env.object_name_map)
    print()

    for i in range(1):
        curr_state = env.get_state()

        # Create an object map (need deep copy because is dict of list?)
        prior_object_names = set(env.OB_NAMES)
        prior_object_names.add("wall")
        prior_object_names.remove("taxi")

        current_object_names = env.get_object_names()
        object_map = {unknown: prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}

        info_gains = []
        for a in range(num_actions):
            info_gain = information_gain_of_action(env, curr_state, a, object_map, previous_ruleset)
            info_gains.append(info_gain)

        info_gain_of_state = information_gain_of_state(env, curr_state, object_map, previous_ruleset)
        print(f"Info gains: {info_gains}")
        print(f"Info gain of state: {info_gain_of_state}")
