"""
Created on 8/2/22 by Ethan Frank


Test "experiment design". What should the agent do in order to figure out
which object are which? What actions should it take
"""


import random
import pickle
import itertools

import numpy as np

from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet
from effects.effect import JointNoEffect
from test.object_transfer.test_object_transfer_learning import determine_possible_object_maps, get_possible_object_assignments

num_actions = 6

env = None


def information_gain_of_action(state: int, action: int, object_map, prev_ruleset: RuleSet) -> float:
    """
    Returns the expected information gain from taking an action, given the current knowledge of the world.
    measured based on net decrease in number of possibilities in object map (wrong, it's better to have one go to 0
    then 3 go down by 1), use entropy total decrease instead)
    """
    known_objects = {"wall", "dest", "pass"}  # TODO: hardcoded for now

    print(f"State {state}")
    print(f"Action {action}")
    print(f"Current object map: {object_map}")

    literals, _ = env.get_literals(state)
    print(f"Literals: {literals}")

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    assert len(applicable_rules) == 1, "My code only works for one rule for now"
    rule = applicable_rules[0]
    assert len(rule.outcomes.outcomes) == 1, "Only deal with one possible outcome"
    print(f"Applicable rules: {rule}")

    # Description of how my brute force algorithm works.
    # Step 1: There are some objects in the state, and some we have the whole list of previously known object
    # Create every combination of mappings possible
    context_objects = set([diectic_obj.split("-")[-1] for diectic_obj in rule.context.referenced_objects])
    state_objects = set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])
    print(f"Context objects: {context_objects}")
    print(f"State objects: {state_objects}")

    print("Permutations:")
    permutations = itertools.permutations(known_objects, len(state_objects))
    for permutation in permutations:
        mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
        print(mapping)
        mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

        new_literals = literals.copy_replace_names(mapping)
        print(f"New tree: {new_literals}")

        applicable = new_literals.base_object.contains(rule.context.base_object)
        print(f"Applicable: {applicable}")

        outcome = Outcome(JointNoEffect())
        if applicable:
            outcome = rule.outcomes.outcomes[0]

        print(f"Predicted outcome: {outcome}")

        # Get object assignments from this example. Is this the part that could be shortcut?
        example = Example(action, literals, outcome)
        possible_assignment = [get_possible_object_assignments(example, previous_ruleset)]
        print(f"Possible assignments: {possible_assignment}")

        print(object_map)
        # Need to use deepcopy on object map because it is a dict of lists
        object_map_copy = {key: value.copy() for key, value in object_map.items()}
        new_object_map = determine_possible_object_maps(object_map_copy, possible_assignment)
        print(f"New object map: {new_object_map}")
        print(f"New total length: {sum(len(possibilites) for possibilites in new_object_map.values())}")

    # Step 2: Assuming that mapping is the real one, see what would happen.

    # Step 3: Given what would happen and what we know, would we gain any information from that action?

    # There is probably a less roundabout method of doing this. For example, maybe we only care about objects
    # That are referenced the way the objects in the rule is referenced?
    # See if the ones where those are the same have same/different effects

    return 0.0


def information_gain_of_state(state: int) -> float:
    """Returns the total info gain over all actions for a state"""
    return sum([information_gain_of_action(state, a) for a in range(num_actions)])


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

    # print("Previous Ruleset")
    # print(previous_ruleset)
    # print()

    possible_assignments = set()

    for i in range(1):
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()

        # Create an object map (need deep copy because is dict of list?)
        prior_object_names = ["pass", "dest", "wall"]
        current_object_names = env.get_object_names()
        object_map = {unknown: prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}

        info_gain = information_gain_of_action(curr_state, 2, object_map, previous_ruleset)


        #
        # literals, observation, name_id_map = env.step(action)
        #
        # print(f"---- Step {i} taking action {action} ----")
        # print(literals)
        # print(observation)
        # print()
        #
        # outcome = Outcome(observation)
        # example = Example(action, literals, outcome)
        # examples.add_example(example)
        #
        # assignments = get_possible_object_assignments(example, previous_ruleset)
        # possible_assignments.add(assignments)
        # print("All assignments: ")
        # print(possible_assignments)
        # print()
