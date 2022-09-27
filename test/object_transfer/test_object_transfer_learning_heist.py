import random
import pickle
from typing import List

import numpy as np

from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet
from test.object_transfer.test_object_transfer_functions import determine_bindings_for_same_outcome, determine_bindings_for_no_outcome
from test.object_transfer.test_object_transfer_functions import ObjectAssignmentList


def get_possible_object_assignments(example: Example, prev_ruleset: RuleSet) -> ObjectAssignmentList:
    """
    Return possible unknown to known object assignments for this example
    and the previously known ruleset
    """

    action = example.action
    literals = example.state
    outcome = example.outcome

    # Get the rules that apply to this situation
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]

    # To determine possible causes, loop over all rules. Some of them might not even apply,
    # So I think we first need to check if they could apply. Otherwise, ignore that rule?
    # Because when there was only one rule we knew that was the one that should apply?
    # But sometimes it didn't? No I think it always did

    # Keeps track of if we found assignments. Only used to verify that when more than one rule is applicable,
    # at least one is matched if the outcomes matched.
    # See determine_bindings_for_same_outcome for more info (when it returns None)
    found_an_assignment = False

    all_assignments = ObjectAssignmentList([])
    print(f"{len(applicable_rules)} applicable rules")
    for rule in applicable_rules:
        # Wait, could the same action have different outcomes depending on the situation?
        outcome_occured = type(outcome.outcome) == type(rule.outcomes.outcomes[0].outcome)

        # Use different reasoning based on if we had a positive or negative example
        if outcome_occured:
            assignments = determine_bindings_for_same_outcome(rule.context, literals)
        else:
            assignments = determine_bindings_for_no_outcome(rule.context, literals)

        print(f"Assignments for\n{rule}:\n{assignments}")
        print()
        if assignments is not None:
            found_an_assignment = True
            all_assignments.add_assignments(assignments)

    assert not (len(applicable_rules) > 1 and not found_an_assignment), "At least one rule must not return None"

    # Make sure all rules have the same outcome. What happens if they don't?
    assert all(type(rule.outcomes.outcomes[0].outcome) ==
               type(applicable_rules[0].outcomes.outcomes[0].outcome)
               for rule in applicable_rules)

    print(f"All possible assignments: {all_assignments}")
    return all_assignments


def determine_possible_object_maps(object_map: dict, possible_assignments: List[ObjectAssignmentList]):
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
            for unknown, knowns in assignment.negatives.items():
                for known in knowns:  # Negatives are list of things the object isn't
                    if known in object_map[unknown]:
                        object_map[unknown].remove(known)

        # If there are two possible assignments, for now, let's just apply all of them
        # If there are multiple positives, say it could be either. If negatives, just remove all of those
        # assert len(assignment_list.assignments) <= 1, "Do not know how to deal with multiple options yet"
        elif len(assignment_list.assignments) > 1:
            # print("Dealing with more than one assigment")
            # print(assignment_list)
            # print()
            # Generate list of positive possibilities for each assignment
            # Narrow down the object map using this list
            positive_possibilities = dict()
            for assignment in assignment_list.assignments:
                for unknown, known in assignment.positives.items():
                    if unknown not in positive_possibilities:
                        positive_possibilities[unknown] = [known]
                    else:
                        positive_possibilities[unknown].append(known)

            # print(positive_possibilities)
            for unknown, knowns in positive_possibilities.items():
                object_map[unknown] = [value for value in object_map[unknown] if value in knowns]

        # If length is 0 we don't have to do anything

    # Check if any objects have been brought to 1, if so, remove those from the others
    # If this causes another to go to one, continue looping
    changed = True
    while changed:
        changed = False
        for known, unknowns in object_map.items():
            if len(unknowns) == 1:
                unknown = unknowns[0]
                for known2, unknowns2 in object_map.items():
                    if known2 != known and unknown in unknowns2:
                        unknowns2.remove(unknown)
                        changed = True

    # Verify no lists went to 0, which indicates a conflict in rules somewhere
    for known, unknowns in object_map.items():
        assert len(unknowns) > 0, f"Should always have a belief about what objects it is: {known}"

    return object_map

# TODO: test multiple rules, properties effects on learning ruleset.


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicHeist(stochastic=False, shuffle_object_names=True)
    env.restart()

    examples = ExampleSet()

    # Load previously learned model with different object names
    with open("../runners/symbolic_heist_rules.pkl", 'rb') as f:
        previous_ruleset = pickle.load(f)

    print("Object name map:")
    print(env.object_name_map)
    print()

    print("Previous Ruleset")
    print(previous_ruleset)
    print()

    possible_assignments = set()

    for i in range(1500):
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

        length_of_knowledge = len(possible_assignments)
        possible_assignments.add(assignments)
        new_length_of_knowledge = len(possible_assignments)
        if new_length_of_knowledge != length_of_knowledge:
            print("Learned Something New")

        print("All assignments: ")
        print(possible_assignments)
        print()

    # Create an object map and then pare it down
    prior_object_names = ["lock", "key", "gem", "wall"]
    current_object_names = env.get_object_names()
    object_map = {unknown: prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}
    new_object_map = determine_possible_object_maps(object_map, possible_assignments)

    print("New object map:")
    for key, value in new_object_map.items():
        print(f"{key}: {value}")
