import random
import time
import pickle
import itertools

import numpy as np

from environment.symbolic_heist import SymbolicHeist
from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from effects.effect import JointNoEffect


def update_object_map_likelihoods(example, prev_ruleset: RuleSet, likelihoods, prior_names, current_names):
    action = example.action
    literals = example.state
    outcome = example.outcome

    # Get the rules that apply to this situation
    print("Applicable rules:")
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    print(applicable_rules)
    print()

    assert len(applicable_rules) == 1, "My code only works for one rule for now"

    # Get the previous objects mentioned in the rule, and try to map them to the objects mentioned in the literals
    print("Referenced objects in literals")
    # literals.referenced_objects returns a list like "taxi0-TOUCH_DOWN2D-idpyo0", we extract just the idpyo part
    # The 0 is actually added separetly and is not really part of the string
    unknown_objects = set(ob.split("-")[-1] for ob in literals.referenced_objects)
    print(unknown_objects)
    print()

    print("Referenced object in known rule")
    known_objects = set(ob.split("-")[-1] for ob in applicable_rules[0].context.referenced_objects)
    print(known_objects)
    print()

    # Generate all possible mappings of object a to objects b
    permutations = itertools.permutations(unknown_objects, len(known_objects))
    # current_object_names_no_taxi = current_object_names.copy()
    # current_object_names_no_taxi.remove("taxi")
    # permutations = itertools.permutations(current_object_names_no_taxi, len(known_objects))

    print("All possible assignments")
    possible_mappings = applicable_rules[0].context.determine_bindings_that_would_match_trees(literals, type(outcome.outcome) == type(applicable_rules[0].outcomes.outcomes[0].outcome))

    # Careful: you can't use this in a for loop because it gets used up.
    # for permutation in permutations:
    #     print({known_ob: unknown_ob for known_ob, unknown_ob in zip(known_objects, permutation)})

    # For each assignment, check whether making the assigment would lead to the result indicated by the rule
    # and that actually occured in the outcome
    # TODO: sometimes the action is obviously irrelvant because the predicates aren't the same. Can we use that knowledge?

    # possible_mappings = []
    # impossible_mappings = []
    # for permutation in permutations:
    #     # Generate a new rule with the permutation by modifying the existing rule's object names
    #     mapping = {known_ob: unknown_ob for known_ob, unknown_ob in zip(known_objects, permutation)}
    #     # Have to keep taxi the same
    #     mapping["taxi"] = "taxi"
    #     new_context = applicable_rules[0].context.copy_replace_names(mapping)
    #     print(f"Object assignment: {mapping}")
    #     print(f"New context: {new_context}")
    #
    #     # Check if this new context matches
    #     matched = literals.base_object.contains(new_context.base_object)
    #
    #     # This handles the case where the rule predicts something should have happened, but nothing happened
    #     # In this case, we need to find an object mapping that caused the rule to fail to apply. If it fails to apply,
    #     # nothing will happen. i.e., the rule says if there's no wall, we move. We find the mapping that says there is a wall
    #     if type(outcome.outcome) is JointNoEffect and type(applicable_rules[0].outcomes.outcomes[0].outcome) is not JointNoEffect:
    #         if matched:
    #             print("Matched!")
    #             impossible_mappings.append(mapping)
    #         else:
    #             # I think there could be more than one correct mapping? I.e., wall and lock
    #             print("No match")
    #             # possible_mappings.append(mapping)
    #             pass
    #     else:
    #         print("Was not a case where something should have happened but didn't")

    print(f"Resulting possible maps: {possible_mappings}")
    # print(f"Resulting impossible maps: {impossible_mappings}")
    # Go through possible mappings and update the likelihood array with bayes rule?
    # Convert names to indecis in the array
    possible_mappings_idx = [{prior_names.index(k): current_object_names.index(v) for k, v in mapping.items()} for mapping in possible_mappings]
    print(f"Resulting possible maps idx: {possible_mappings_idx}")

    # Iterate over mappings, assign a value of one for that index (clear out row and column)
    # What about conditional probabilities?

    # TODO: I don't even know if this is correct because of the conditional probability issue
    # Keep track of possible mappings for each object
    possibilities = {k: [] for mapping in possible_mappings_idx for k in mapping.keys()}
    for mapping in possible_mappings_idx:
        for k, v in mapping.items():
            possibilities[k].append(v)

    print(f"Possibilities {possibilities}")
    # Using the (independent) possibilites, update the likelihoods
    for k, value in possibilities.items():
        likelihood = [0] * likelihoods.shape[1]
        for ob in value:
            likelihood[ob] = 1

        likelihoods[k] *= likelihood

    # Need to have rows sum to 1, so need to transpose to get the alignment
    likelihoods = (likelihoods.T / np.sum(likelihoods, axis=1)).T
    # Need to update columns if any row becomes determinant, since only one object can be assigned a thing
    for k in possibilities.keys():
        best_guess = np.argmax(likelihoods[k])

        # If we are certain that is the correct guess, wipe that column except for that one
        # Then update the rows to still sum to one
        if likelihoods[k, best_guess] == 1:
            likelihoods[:, best_guess] = 0
            likelihoods[k, best_guess] = 1

    # Need to have rows sum to 1, so need to transpose to get the alignment
    likelihoods = (likelihoods.T / np.sum(likelihoods, axis=1)).T

    # assert len(possible_mappings) <= 1, f"Can only handle one possible mapping right now, not {len(possible_mappings)}"
    assert len(possibilities) <= 1, "Should only be one object in the possibilities array, conditional probability"
    # Then, normalize likelihoods to sum to one.

    return likelihoods


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

    prior_object_names = ["taxi", "pass", "dest", "wall"]
    current_object_names = env.get_object_names()

    # Assign initial probability for each object being each other object
    object_map_belief = np.ones((len(prior_object_names), len(current_object_names)))

    object_map_belief /= (len(current_object_names) - 1)
    object_map_belief[0, 1:] = 0  # Set taxi to be known as taxi
    object_map_belief[1:, 0] = 0
    object_map_belief[0, 0] = 1
    print("Starting likelihood map:")
    print(object_map_belief)
    print()

    for i in range(1000):
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()

        literals, observation, name_id_map = env.step(action)

        print(f"Step {i} taking action {action}")
        print(literals)
        # print(name_id_map)
        print(observation)
        print()

        outcome = Outcome(observation)
        example = Example(action, literals, outcome)
        examples.add_example(example)

        # if i == 0:  # Only want to test second action here
        #     continue
        print("Updating likelihoods")
        object_map_belief = update_object_map_likelihoods(
            example, previous_ruleset, object_map_belief, prior_object_names, current_object_names
        )
        print("New likelihoods")
        print(object_map_belief)
        print()

    # print("Examples")
    # print(examples)
    # print()
