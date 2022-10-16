"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""
import random
import numpy as np
import time
import graphviz

from environment.symbolic_heist import SymbolicHeist
from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, Rule, OutcomeSet
from symbolic_stochastic_domains.predicates_and_objects import In, Open, TouchDown, PredicateType, On
from symbolic_stochastic_domains.symbolic_utils import applicable, context_matches
from symbolic_stochastic_domains.predicate_tree import PredicateTree, Edge, Node

from effects.effect import JointNoEffect

# import cProfile
# import pstats

from test.test_predicate_tree import plot_predicate_tree


def update_experience_dict(experience: dict, example: Example):
    # Experience dict is a list of how many times we have tried for every object, every way to interacti with that
    # object, for every action, how many times we've tried each

    # To start with, only look at objects connected to the base object, taxi
    # TODO: tuples instead of nested dicts?
    for edge in example.state.base_object.edges:
        to_object = edge.to_node.object_name[:-1]

        if to_object not in experience:
            experience[to_object] = dict()

        edge_type = str(edge.type)[14:] + ("_OPEN_" + str(edge.to_node.properties[PredicateType.OPEN]) if len(edge.to_node.properties) > 0 else "")
        if edge_type not in experience[to_object]:
            experience[to_object][edge_type] = dict()

        if example.action not in experience[to_object][edge_type]:
            experience[to_object][edge_type][example.action] = 1
        else:
            experience[to_object][edge_type][example.action] += 1


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    # env = SymbolicHeist(stochastic=False)
    env = SymbolicTaxi(stochastic=False, shuffle_object_names=True)
    env.restart()  # The env is being restarted twice in the runner, which means random key arrangements were different

    examples = ExampleSet()

    experience = dict()

    print(env.object_name_map)

    # for action in actions:
    for i in range(1000):  # This breaks at 1130, due to trying to go down while touching a door and 3061
        action = random.randint(0, env.get_num_actions()-1)
        curr_state = env.get_state()
        # literals = env.get_literals(curr_state)
        literals, observation, name_id_map = env.step(action)  # , predicate_to_ob_map, obs_grounding

        # if i > 3110:
        print(literals)
        print(name_id_map)
        print(observation)
        print()

            # graph = graphviz.Digraph(format='png')
            # # graph.engine = 'neato'
            # # graph.graph_attr.update(nodesep="5")
            # plot_predicate_tree(literals, graph)
            # graph.view()
            #
        # env.draw_world(curr_state, delay=1)

        outcome = Outcome(observation)
        example = Example(action, literals, outcome)
        examples.add_example(example)

        update_experience_dict(experience, example)
        # print(f"Experience: {experience}")
        if i == 1000:
            for key, value in experience.items():
                print(key, value)

    # print("Examples")
    # print(examples)
    # print()

    # Why don't I just literally represent them as graphs? It wouldn't be that hard.
    # I could even do just a bunch of nested dictionaries. I think I want to try that next.
    # Object are the nodes, predicates are the edges, and objects have properties.

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

    # print(f"Context: {context}")
    # print(context.node_lookup["lock0"])
    # for example in examples.examples:
    #     if context_matches(context, example.state):
    #         print(f"Context matched {example}")
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
    start_time = time.perf_counter()
    # for i in range(10):
    # profiler = cProfile.Profile()
    # profiler.enable()
    solver = RulesetLearner(env)
    ruleset = solver.learn_ruleset(examples)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('stats.prof')
    end_time = time.perf_counter()
    print(f"Took {end_time-start_time}")
    print("Resulting ruleset:")
    print(ruleset)

    # for i, rule in enumerate(ruleset.rules):
    #     graph = graphviz.Digraph(name=str(rule), format='png')
    #     plot_predicate_tree(rule.context, graph)
    #     graph.view()
