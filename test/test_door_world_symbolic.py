import logging
import random
import numpy as np

from environment.symbolic_door_world import SymbolicDoorWorld, TouchLeft, Taxi, Switch
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet, Rule, OutcomeSet, Outcome
from effects.effect import JointEffect, Increment, SetToNumber, JointNoEffect
from symbolic_stochastic_domains.symbolic_utils import context_matches, examples_covered_by_outcome

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicDoorWorld()

    print(env.visualize())

    example_set = ExampleSet()

    for i in range(5):
        literals = env.get_literals(env.curr_state)
        action = random.randint(0, env.get_num_actions()-1)

        obs = env.step(action)
        #
        # print(f"Took action {action}")
        # print(env.visualize())
        # print(f"Observed {obs}")

        example = Example(action, literals, obs)

        # print("Training example:")
        # print(example)
        # print("-----")

        example_set.add_example(example)

    print("Example set:")
    print(example_set)
    print()

    action = 1
    # context = [TouchLeft(Taxi(x=3, name="taxi"), Switch(x=2, name="switch"))]
    context = []
    outcomes = OutcomeSet()
    # outcomes.add_outcome(Outcome(JointEffect(["taxi.x", "door.open"], [Increment(1, 0), SetToNumber(False, True)])), 1.0)
    outcomes.add_outcome(Outcome(JointEffect(["taxi.x"], [Increment(1, 0)])), 1.0)
    # outcomes.add_outcome(JointNoEffect(), 0.5)

    rule = Rule(0, context, outcomes)

    print("Rule 1:")
    print(rule)
    score = rule.score(example_set)
    print(f"Score: {score:0.2}")
    print()

    print("Checking for context match")
    print(f"Context: {rule.context}")
    for example in example_set.examples.keys():
        matches = context_matches(rule.context, example.state)
        print(f"Example {example}")
        print(f"Matches: {matches}")
    print()

    covered_examples = examples_covered_by_outcome(rule.outcomes.outcomes[0], example_set)
    print("Covered examples:")
    for example in covered_examples:
        print(example)


