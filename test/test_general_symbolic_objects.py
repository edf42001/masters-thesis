"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""

import random
import numpy as np
import time

from environment.symbolic_heist import SymbolicHeist
from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, Rule, OutcomeSet
from symbolic_stochastic_domains.predicates_and_objects import In, Open, TouchDown, PredicateType, On
from symbolic_stochastic_domains.symbolic_utils import applicable, context_matches
from symbolic_stochastic_domains.predicate_tree import PredicateTree, Edge, Node

import cProfile
import pstats


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicHeist(stochastic=False)
    # env = SymbolicTaxi(stochastic=False, shuffle_object_names=False)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    examples = ExampleSet()

    experience = dict()

    # print(env.object_name_map)

    # for action in actions:
    for i in range(1000):  # This breaks at 1130, due to trying to go down while touching a door and 3061
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        # literals = env.get_literals(curr_state)
        literals, outcome, name_id_map = env.step(action)  # , predicate_to_ob_map, obs_grounding

        # print(literals)
        # print(name_id_map)
        # print(outcome)
        # print()

        # env.draw_world(curr_state, delay=1)

        example = Example(action, literals, outcome)
        examples.add_example(example)

    # print("Examples")
    # print(examples)
    # print()

    # Use tree learning. When in env, only need to update the outcome that was changed.
    # Hm, actually I don't know if that's true. Perhaps save the values for each thing, and update like that?
    # Also, run only on certain action.

    # outcomes = OutcomeSet()
    # outcomes.add_outcome(Outcome(JointNoEffect()), 1.0)
    # test_rule = Rule(action=5, context=context, outcomes=outcomes)

    # Rule: if a positive literal is in the context, it must be referred to in the deictic references
    # But wait, aren't the references kindof similar to the context itself? They force matches to be found,
    # If you put something in there it acts as context. Maybe I should just make my graph? Might be hard to learn
    # context = PredicateTree()
    # context.add_node("taxi0")
    # context.add_node("lock0")
    # context.add_edge("taxi0", "lock0", PredicateType.TOUCH_DOWN2D)
    # context.add_property("lock0", PredicateType.OPEN, True)
    # context.add_edge("taxi0", "key1", PredicateType.IN, negative=True)
    # context.base_object.add_edge(Edge(PredicateType.ON2D, Node("key")))
    # context.base_object.add_negative_edge(Edge(PredicateType.IN, Node("key")))

    start_time = time.perf_counter()
    # for i in range(10):
    # profiler = cProfile.Profile()
    # profiler.enable()
    solver = RulesetLearner()
    ruleset = solver.learn_ruleset(examples)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('ruleset_learning_stats_1.prof')
    end_time = time.perf_counter()
    print(f"Took {end_time-start_time}")
    print("Resulting ruleset:")
    print(ruleset)
