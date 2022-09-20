import random
import pickle

import numpy as np

from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet
from test.object_transfer.test_object_transfer_learning_heist import get_possible_object_assignments, determine_possible_object_maps


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicTaxi(stochastic=False, shuffle_object_names=True)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    examples = ExampleSet()

    # Load previously learned model with different object names
    with open("../runners/symbolic_taxi_rules.pkl", 'rb') as f:
        previous_ruleset = pickle.load(f)

    print("Object name map:")
    print(env.object_name_map)
    print()

    print("Previous Ruleset")
    print(previous_ruleset)
    print()

    possible_assignments = set()

    for i in range(335):
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()

        literals, observation, name_id_map = env.step(action)

        print(f"---- Step {i} taking action {action} ----")
        print(literals)
        print(observation)
        print()

        outcome = Outcome(observation)
        example = Example(action, literals, outcome)
        examples.add_example(example)

        assignments = get_possible_object_assignments(example, previous_ruleset)
        possible_assignments.add(assignments)
        print("All assignments: ")
        print(possible_assignments)
        print()

    # Create an object map and then pare it down
    prior_object_names = ["pass", "dest", "wall"]
    current_object_names = env.get_object_names()
    object_map = {unknown: prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}
    new_object_map = determine_possible_object_maps(object_map, possible_assignments)

    print("New object map:")
    for key, value in new_object_map.items():
        print(f"{key}: {value}")
