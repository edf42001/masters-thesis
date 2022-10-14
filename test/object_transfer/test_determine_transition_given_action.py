"""
Created on 9/1/22 by Ethan Frank

Tests with determine_next_state_given action given various object maps
"""

import random
import pickle

import numpy as np

from environment.symbolic_taxi import SymbolicTaxi
from environment.symbolic_heist import SymbolicHeist

from symbolic_stochastic_domains.object_transfer import determine_transition_given_action

num_actions = 6

random.seed(1)
np.random.seed(1)

env = SymbolicHeist(stochastic=False, shuffle_object_names=True)
env.restart()


if __name__ == "__main__":
    # Load previously learned model with different object names
    with open("../runners/symbolic_heist_rules.pkl", 'rb') as f:
        previous_ruleset = pickle.load(f)

    # Current object map belief
    object_map = {'idpyo': ['key'],
                  'pumzg': ['lock'],
                  'dpamn': ['gem'],
                  'tyyaw': ['wall', 'lock']}

    # env.restart(init_state=env.get_factored_state(19440))
    curr_state = env.get_state()
    literals, _ = env.get_literals(curr_state)

    print(literals)
    action = 2
    transition = determine_transition_given_action(env, curr_state, action, object_map, previous_ruleset)

    print(f"Transition: {transition}")
