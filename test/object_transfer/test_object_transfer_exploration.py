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
from test.object_transfer.test_object_transfer_learning_heist import determine_possible_object_maps, get_possible_object_assignments

num_actions = 6


def information_gain_of_action(env, state: int, action: int, object_map, prev_ruleset: RuleSet) -> float:
    """
    Returns the expected information gain from taking an action, given the current knowledge of the world.
    measured based on net decrease in number of possibilities in object map (wrong, it's better to have one go to 0
    then 3 go down by 1), use entropy total decrease instead)
    """
    # List of known object names without taxi and with wall (wall is static so is not in the list normally)
    known_objects = set(env.OB_NAMES)
    known_objects.add("wall")
    known_objects.remove("taxi")

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

    # Track total info gain and number of permutations so we can calculate an expected info gain
    total_info_gain = 0
    num_permutations = 0

    mappings_to_choose_from = (object_map[unknown_object] for unknown_object in state_objects)
    permutations = itertools.product(*mappings_to_choose_from)

    # permutations = itertools.permutations(known_objects, len(state_objects))
    print("Permutations:")
    for permutation in permutations:
        # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
        # Technically there should be no reason why not but it breaks literals.copy_replace_names
        if len(set(permutation)) != len(permutation):
            continue

        num_permutations += 1
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
        possible_assignment = [get_possible_object_assignments(example, prev_ruleset)]
        print(f"Possible assignments: {possible_assignment}")

        print(object_map)

        new_object_map = determine_possible_object_maps(object_map, possible_assignment)
        prev_num_options = sum(len(possibilities) for possibilities in object_map.values())
        new_num_options = sum(len(possibilities) for possibilities in new_object_map.values())

        print(f"New object map: {new_object_map}")
        print(f"Length update: {prev_num_options}->{new_num_options}")

        # Info gain is change in bits required to express number of object possibilities, which is log2 of length
        total_info_gain += np.log2(prev_num_options) - np.log2(new_num_options)

    # Step 2: Assuming that mapping is the real one, see what would happen.

    # Step 3: Given what would happen and what we know, would we gain any information from that action?

    # There is probably a less roundabout method of doing this. For example, maybe we only care about objects
    # That are referenced the way the objects in the rule is referenced?
    # See if the ones where those are the same have same/different effects
    print(f"Total info {total_info_gain}, num permutations: {num_permutations}")
    return total_info_gain / num_permutations


def information_gain_of_state(env, state: int, object_map, prev_ruleset: RuleSet) -> float:
    """Returns the total info gain over all actions for a state"""
    return sum([information_gain_of_action(env, state, a, object_map, prev_ruleset) for a in range(num_actions)])


def determine_transition_given_action(env, state: int, action: int, object_map, prev_ruleset: RuleSet):
    """
    Given a current state and current object map belief,
    what are the possible next states for a specific actions?
    If there is more than one possibility depending on object map bindings, return None.
    This is an opportunity to gain information (possibly?). Otherwise, return the transition? Or the state?
    """

    # All this is the exact same as information_gain_of_action. See there for more info.
    # In fact, all this code is the exact same, because in order to know what info we learn we
    # have to figure out what the transition was base on the previous rule.

    literals, _ = env.get_literals(state)

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    assert len(applicable_rules) == 1, "My code only works for one rule for now"
    rule = applicable_rules[0]
    assert len(rule.outcomes.outcomes) == 1, "Only deal with one possible outcome"

    # TODO: Exclude permutations that are not relavent to the rule, or not relevant to the current object map
    context_objects = set([diectic_obj.split("-")[-1] for diectic_obj in rule.context.referenced_objects])

    # Objects that are currently in the state
    state_objects = set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])

    applicable_tracker = None  # Stores outcome as we process permutations so we can look for contradictions

    # Filter the object map by only objects in the state. Then, create all possible combinations of state objects
    # and what we believe they could be. Then, check if all the outcomes match.
    mappings_to_choose_from = (object_map[unknown_object] for unknown_object in state_objects)
    permutations = itertools.product(*mappings_to_choose_from)

    for permutation in permutations:
        # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
        # Technically there should be no reason why not but it breaks literals.copy_replace_names
        if len(set(permutation)) != len(permutation):
            continue

        mapping = {state_object: permute_object for state_object, permute_object in zip(state_objects, permutation)}
        mapping["taxi"] = "taxi"  # Taxi has to be there but always maps to itself

        new_literals = literals.copy_replace_names(mapping)

        applicable = new_literals.base_object.contains(rule.context.base_object)

        if applicable_tracker is None:
            applicable_tracker = applicable
        elif applicable_tracker != applicable:  # If we every get different answers, we don't know so return none
            return None

    # Returns the outcome if something will happen, no effect if nothing was applicable to the rule
    # TODO: This returns the object names from the rule. It should probably replace those with the current object names
    return rule.outcomes.outcomes[0] if applicable_tracker else Outcome(JointNoEffect())


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    examples = ExampleSet()

    # These will eventually be stored in a class as a member variable for easy access
    env = SymbolicTaxi(stochastic=False, shuffle_object_names=True)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    # Load previously learned model with different object names
    with open("../runners/symbolic_taxi_rules.pkl", 'rb') as f:
        previous_ruleset = pickle.load(f)

    print("Object name map:")
    print(env.object_name_map)
    print()

    for i in range(1):
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()

        # Create an object map (need deep copy because is dict of list?)
        prior_object_names = ["pass", "dest", "wall"]
        current_object_names = env.get_object_names()
        object_map = {unknown: prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}

        info_gains = []
        for a in range(num_actions):
            info_gain = information_gain_of_action(env, curr_state, a, object_map, previous_ruleset)
            info_gains.append(info_gain)

        info_gain_of_state = information_gain_of_state(env, curr_state, object_map, previous_ruleset)
        print(f"Info gains: {info_gains}")
        print(f"Info gain of state: {info_gain_of_state}")
