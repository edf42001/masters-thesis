"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""
import random
import numpy as np
import time

from environment.symbolic_heist import SymbolicHeist
from symbolic_stochastic_domains.learn_ruleset_outcomes import learn_ruleset_outcomes
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, Rule, OutcomeSet
from symbolic_stochastic_domains.predicates_and_objects import In, Open, TouchDown2D, PredicateType, On2D
from symbolic_stochastic_domains.symbolic_utils import applicable, context_matches
from symbolic_stochastic_domains.predicate_tree import PredicateTree, Edge, Node

from effects.effect import JointNoEffect

# import cProfile
# import pstats


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicHeist(stochastic=False)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    examples = ExampleSet()

    actions = [env.A_PICKUP, env.A_WEST, env.A_PICKUP]
    # for action in actions:
    for i in range(3061):  # This breaks at 1130, due to trying to go down while touching a door
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        # literals = env.get_literals(curr_state)
        literals, observation, name_id_map = env.step(action)  # , predicate_to_ob_map, obs_grounding

        if i > 3058:
            print(literals)
            print(name_id_map)
            print(observation)
            print()
        # env.draw_world(curr_state, delay=5)

        outcome = Outcome(observation)
        example = Example(action, literals, outcome)
        examples.add_example(example)

    # print("Examples")
    # print(examples)
    # print()

    # Why don't I just literally represent them as graphs? It wouldn't be that hard.
    # I could even do just a bunch of nested dictionaries. I think I want to try that next.
    # Object are the nodes, predicates are the edges, and objects have properties.

    # Rule: if a positive literal is in the context, it must be referred to in the deictic references
    # But wait, aren't the references kindof similar to the context itself? They force matches to be found,
    # If you put something in there it acts as context. Maybe I should just make my graph? Might be hard to learn
    # context = PredicateTree()
    # context.base_object.add_edge(Edge(PredicateType.ON2D, Node("key")))
    # context.base_object.add_negative_edge(Edge(PredicateType.IN, Node("key")))

    # Use tree learning. When in env, only need to update the outcome that was changed.
    # Hm, actually I don't know if that's true. Perhaps save the values for each thing, and update like that?
    # Also, run only on certain action.

    # outcomes = OutcomeSet()
    # outcomes.add_outcome(Outcome(JointNoEffect()), 1.0)
    # test_rule = Rule(action=5, context=context, outcomes=outcomes)

    # print(f"Context: {context}")
    # for example in examples.examples:
    #     if context_matches(context, example.state):
    #         print(f"Context matched {example}")
        # else:
        #     print(f"Context did not match {example}")

    # The issue is that the taxi can not learn the correct ruleset, because we just have Open(lock, lock)
    # so when it touches the nonopen lock and tries to go down it doesn't, which means there's a conflict, because
    # it doesn't know which lock is open. Need to use properties. Or, just need to know which lock which predicate
    # refers to. So, a mapping from TouchDown(lock), Open(lock), OnLock(closed, lock)
    # So for each object mentioned in the state, return it's properties (or even it's predicates?). (in a dict/list?)
    # OK, the bindings are a good start, but that information would have to go into the example, since it is part of the state
    # Can I just store only positive literals in the example state and assume everything else is negative??
    # Then I would need a full list of all potential literals in a state somewhere.
    # Avoid infinite cycles by making sure an object can reference an object that references it.
    # I think I like the idea of predicates refering to specific object ids, as well as the class
    start_time = time.perf_counter()
    # for i in range(10):
    # profiler = cProfile.Profile()
    # profiler.enable()
    # ruleset = learn_ruleset_outcomes(examples)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('stats.prof')
    end_time = time.perf_counter()
    print(f"Took {end_time-start_time}")
    print("Resulting ruleset:")
    # print(ruleset)
