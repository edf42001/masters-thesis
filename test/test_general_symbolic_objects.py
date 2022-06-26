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
from symbolic_stochastic_domains.symbolic_utils import applicable, context_matches, applies_with_deictic
from symbolic_stochastic_domains.learn_outcomes import learn_outcomes

from effects.effect import JointNoEffect


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    env = SymbolicHeist(stochastic=False, use_outcomes=False)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    examples = ExampleSet()

    actions = [env.A_PICKUP, env.A_WEST, env.A_PICKUP]
    # for action in actions:
    for i in range(1130):  # This breaks at 1130, due to trying to go down while touching a door
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        literals = env.get_literals(curr_state)
        observation = env.step(action)  # , predicate_to_ob_map, obs_grounding

        # if i > 1120:
        literals.print()
        # print(ob_id_name_map)
        print(observation)
        print()
        # env.draw_world(curr_state, delay=700)

    #     outcome = Outcome(observation)
    #     example = Example(action, literals, outcome)
    #     examples.add_example(example)
    #
    # print("Examples")
    # print(examples)
    # print()
    # Why don't I just literally represent them as graphs? It wouldn't be that hard.
    # I could even do just a bunch of nested dictionaries. I think I want to try that next.
    # Object are the nodes, predicates are the edges, and objects have properties.

    # Rule: if a positive literal is in the context, it must be referred to in the deictic references
    # But wait, aren't the references kindof similar to the context itself? They force matches to be found,
    # If you put something in there it acts as context. Maybe I should just make my graph? Might be hard to learn
    deictic_references = {"key3": On2D(PredicateType.ON2D, "taxi0", "key3", True)}
    context = [In(PredicateType.IN, "taxi0", "key4", False)]

    print("Applicable to:")
    for example in examples.examples:
        if applies_with_deictic(deictic_references, context, example.state):
            print(f"Was applicable to {example}")
        # else:
            # print(f"Context did not match {example}")

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
    # for ex in example_set.examples:
    #     print(ex)
    # start_time = time.perf_counter()
    # # for i in range(10):
    # ruleset = learn_ruleset_outcomes(example_set)
    # end_time = time.perf_counter()
    # print(f"Took {end_time-start_time}")
    # print("Resulting ruleset:")
    # print(ruleset)
    #
    # relevant_examples = [example for example in example_set.examples.keys() if example.action == 4 and type(example.outcome.outcome) is JointNoEffect]
    #
    # print("Relevant examples:")
    # for example in relevant_examples:
    #     print(example)
    #
    # test_rule = Rule(action=2, references={TouchDown2D(PredicateType.TOUCH_DOWN2D, "taxi", "door", True): Open(PredicateType.OPEN, "door", "door", True)},
    #                  context=[TouchDown2D(PredicateType.TOUCH_DOWN2D, "taxi", "door", True),
    #                           Open(PredicateType.OPEN, "door", "door", True)], outcomes=OutcomeSet())
