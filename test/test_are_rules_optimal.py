import random
import time
import pickle

from environment.symbolic_door_world import SymbolicDoorWorld, TouchLeft, Taxi, Switch
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, OutcomeSet, Outcome, RuleSet
from symbolic_stochastic_domains.learn_ruleset import learn_ruleset, calculate_default_rule
from symbolic_stochastic_domains.symbolic_utils import ruleset_score, rule_score, examples_applicable_by_rule


def print_examples_rules_cover(ruleset: RuleSet, examples: ExampleSet):
    for i, rule in enumerate(ruleset.rules):
        if i == 0:
            applicable = ruleset.default_rule_covered_examples
        else:
            applicable = examples_applicable_by_rule(rule, examples)
        print(f"Rule {i}: {applicable}")


if __name__ == "__main__":
    # I found that in python, hashes are seeded with a random seed.
    # So I can't generate examples like this and use a loaded ruleset, because the
    # literals in the ruleset will have different hashes. This can be turned off with an environment
    # variable or by using hashlib
    
    random.seed(1)

    # Read ruleset and examples instead of regenerating to save cpu
    with open("deterministic_300_steps_door_world_examples_rules.pkl", 'rb') as f:
        examples, ruleset = pickle.load(f)

    print(examples)
    print()
    print(ruleset)
    print()

    start_score = ruleset_score(ruleset, examples)

    print(start_score)
    print()
    print("Rules cover: ")
    print_examples_rules_cover(ruleset, examples)

    ruleset2 = ruleset.copy()
    ruleset2.rules.pop(4)
    calculate_default_rule(ruleset2, examples)

    print(ruleset2)
    print("Rules rcover")
    print_examples_rules_cover(ruleset2, examples)
