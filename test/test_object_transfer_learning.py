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
from test.test_object_transfer_functions import determine_bindings_for_same_outcome, determine_bindings_for_no_outcome


def get_possible_object_assignments(example: Example, prev_ruleset: RuleSet):
    """
    Return possible unknown to known object assignments for this example
    and the previously known ruleset
    """

    action = example.action
    literals = example.state
    outcome = example.outcome

    # Get the rules that apply to this situation
    print("Applicable rules:")
    applicable_rules = [rule for rule in prev_ruleset.rules if rule.action == action]
    print(applicable_rules)
    print()

    assert len(applicable_rules) == 1, "My code only works for one rule for now"

    rule = applicable_rules[0]

    outcome_occured = type(outcome.outcome) == type(rule.outcomes.outcomes[0].outcome)

    # Use different reasoning based on if we had a positive or negative example
    if outcome_occured:
        assignments = determine_bindings_for_same_outcome(rule.context, literals)
    else:
        assignments = determine_bindings_for_no_outcome(rule.context, literals)

    print("Assignments were: ")
    print(assignments)

    return assignments


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

    possible_assignments = set()

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

        assignments = get_possible_object_assignments(example, previous_ruleset)
        possible_assignments.add(assignments)
        print("All assignments: ")
        print(possible_assignments)
        print()


    # print("Examples")
    # print(examples)
    # print()
