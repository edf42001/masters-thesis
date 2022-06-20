"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""
import random
import numpy as np

from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.learn_ruleset_outcomes import learn_ruleset_outcomes
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, Rule, OutcomeSet
from symbolic_stochastic_domains.predicates_and_objects import On2D, PredicateType
from symbolic_stochastic_domains.symbolic_utils import applicable
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes

from effects.effect import JointNoEffect


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicHeist(stochastic=False, use_outcomes=False)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    # actions = [env.A_NORTH, env.A_SOUTH, env.A_EAST, env.A_WEST, env.A_NORTH, env.A_PICKUP, env.A_UNLOCK,
    #            env.A_SOUTH, env.A_EAST, env.A_NORTH, env.A_NORTH, env.A_WEST, env.A_WEST,
    #            env.A_SOUTH, env.A_UNLOCK, env.A_SOUTH]

    example_set = ExampleSet()

    # for action in actions:
    for i in range(1120):  # This breaks at 1130, due to trying to go down while touching a door
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        observation = env.step(action)
        literals, bindings = env.get_literals(curr_state)

        # env.draw_world(curr_state, delay=1)

        outcome = Outcome(observation)
        example = Example(action, literals, outcome)
        example_set.add_example(example)

    ruleset = learn_ruleset_outcomes(example_set)
    print("Resulting ruleset:")
    print(ruleset)

    relevant_examples = [example for example in example_set.examples.keys() if example.action == 4 and type(example.outcome.outcome) is JointNoEffect]

    print("Relevant examples:")
    for example in relevant_examples:
        print(example)

    test_rule = Rule(action=4, context=[On2D(PredicateType.ON2D, "taxi", "key", False)], outcomes=OutcomeSet())

    learn_outcomes(test_rule, example_set)
    print("Resulting test rule:")
    print(test_rule)

    # best_rule = find_greedy_rule_by_removing_lits(example_set, relevant_examples, [])
    # print("Best rule: ")
    # print(best_rule)
