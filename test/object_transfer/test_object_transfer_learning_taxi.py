import random
import pickle

import numpy as np

from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet
from test.object_transfer.test_object_transfer_functions import determine_bindings_for_same_outcome, determine_bindings_for_no_outcome


def get_possible_object_assignments(example: Example, prev_ruleset: RuleSet):
    """
    Return possible unknown to known object assignments for this example
    and the previously known ruleset
    """

    action = example.action
    literals = example.state
    outcome = example.outcome

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    assert len(applicable_rules) == 1, "My code only works for one rule for now"

    rule = applicable_rules[0]

    outcome_occured = type(outcome.outcome) == type(rule.outcomes.outcomes[0].outcome)

    # Use different reasoning based on if we had a positive or negative example
    if outcome_occured:
        assignments = determine_bindings_for_same_outcome(rule.context, literals)
    else:
        assignments = determine_bindings_for_no_outcome(rule.context, literals)

    return assignments

# def apply_assignment(object_map, assignment)


def determine_possible_object_maps(object_map, possible_assignments):
    """
    Tries to figure out which assignments are the true ones and which are not,
    in the process learning which object is which
    """

    # Create a copy so as to not modify the original object
    # Need to use deepcopy on object map because it is a dict of lists
    object_map = {key: value.copy() for key, value in object_map.items()}

    # Basically, one or more assignment in each assignment list must be true, find which
    for assignment_list in possible_assignments:
        # If the length is one, then we know that one must be true, so we can apply it
        if len(assignment_list.assignments) == 1:
            assignment = assignment_list.assignments[0]

            # If we know it is one thing, set all of those to their correct value
            for unknown, known in assignment.positives.items():
                assert known in object_map[unknown], f"We say it must be true so it better be an option: {unknown}"
                object_map[unknown] = [known]

            # Remove everything from negatives (it may have been removed already)
            for unknown, known in assignment.negatives.items():
                if known in object_map[unknown]:
                    object_map[unknown].remove(known)

        assert len(assignment_list.assignments) <= 1, "Do not know how to deal with multiple options yet"

    return object_map


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
