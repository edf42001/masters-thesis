"""
Created on 7/30/22 by Ethan Frank

Test low level functions for object knowledge transfer with specific cases
"""


import random
import time
import pickle
import itertools

import numpy as np

from environment.symbolic_heist import SymbolicHeist
from environment.symbolic_taxi import SymbolicTaxi
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Outcome, Example, RuleSet, Rule
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from effects.effect import JointNoEffect
from symbolic_stochastic_domains.predicates_and_objects import PredicateType


class ObjectAssignment:
    """
    Keeps track of one possible assignment from unknown to known
    objects that would match the current observation
    """

    def __init__(self):
        self.positives = {}
        self.negatives = {}

    def add_positive(self, unknown, known):
        self.positives[unknown] = known

    def add_negative(self, unknown, known):
        self.negatives[unknown] = known

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if len(self.positives) > 0 and len(self.negatives) > 0:
            return str(self.positives) + " - ~" + str(self.negatives)
        elif len(self.negatives) > 0:
            return "~" + str(self.negatives)
        elif len(self.positives) > 0:
            return str(self.positives)
        else:
            return "{}"

    def __eq__(self, other):
        return self.positives == other.positives and self.negatives == other.negatives


class ObjectAssignmentList:
    """
    A collection of object assignments retrieved from an example.
    At least one, possibly more, of the assignments must be the true cause
    """

    def __init__(self, assignments):
        self.assignments = assignments
        self.hash = hash(self.__str__())

    def __str__(self):
        return str(self.assignments)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.assignments, other.assignments))


def determine_bindings_for_same_outcome(condition: PredicateTree, state: PredicateTree):
    """
    Say we have a tree representing the condition for a rule.
    The other tree is a state, with different object names.
    What mappings of objects names make it so the rule condition tree is applicable to the other?

    This is for the specific case when the rule says an outcome and the actual outcome matched.
    """

    assignment = ObjectAssignment()

    # First, check all positive edges. All of these must match
    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same
                assignment.add_positive(edge2.to_node.object_name, edge.to_node.object_name)
                found = True
                break

        assert found, "Matching positive edge was not found"

    # Now all negative edges, any matches here must be false
    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment.add_negative(edge2.to_node.object_name, edge.to_node.object_name)
                break

    return ObjectAssignmentList([assignment])


def determine_bindings_for_no_outcome(condition: PredicateTree, state: PredicateTree):
    """
    Say we have a tree representing the condition for a rule.
    The other tree is a state, with different object names.
    What mappings of objects names make it so the rule condition tree is applicable to the other?

    This is for the specific case when the rule says an outcome but instead nothing happens.
    """

    assignments = []

    # First, make sure all the positive edges match. If they don't we can't really give a reason,
    # because the missing positive edge could be why nothing happened
    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same
                found = True
                break

        if not found:
            return ObjectAssignmentList([ObjectAssignment()])

    # Now, let's look for negative edges that perhaps are causing the issue
    # Also some positive edges could be missing. Each of these are independent, add them to the array separately.
    for edge in condition.base_object.edges:
        for edge2 in state.base_object.edges:  # (Could probably combine this with the above, then throw out if bad?)
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_negative(edge2.to_node.object_name, edge.to_node.object_name)
                assignments.append(assignment)
                break

    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_positive(edge2.to_node.object_name, edge.to_node.object_name)
                assignments.append(assignment)
                break

    return ObjectAssignmentList(assignments)


def test_positive_positive():
    # The rule's condition
    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obj1")
    condition.add_node("obj2")
    condition.add_node("obj3")
    condition.add_node("obj4")

    condition.add_edge("taxi0", "obj1", PredicateType.TOUCH_LEFT)
    condition.add_edge("taxi0", "obj2", PredicateType.TOUCH_RIGHT)
    condition.add_edge("taxi0", "obj3", PredicateType.ON, negative=True)
    condition.add_edge("taxi0", "obj4", PredicateType.IN, negative=True)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a")
    state.add_node("b")
    state.add_node("c")
    state.add_node("d")
    state.add_node("e")
    state.add_node("f")

    state.add_edge("taxi0", "a", PredicateType.TOUCH_UP2D)
    state.add_edge("taxi0", "b", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d", PredicateType.TOUCH_DOWN2D)
    state.add_edge("taxi0", "e", PredicateType.ON, negative=False)
    state.add_edge("taxi0", "f", PredicateType.IN, negative=False)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_same_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")


def test_positive_negative():
    # The rule's condition
    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obj1")
    condition.add_node("obj2")
    condition.add_node("obj3")
    condition.add_node("obj4")

    condition.add_edge("taxi0", "obj1", PredicateType.TOUCH_LEFT)
    condition.add_edge("taxi0", "obj2", PredicateType.TOUCH_RIGHT)
    condition.add_edge("taxi0", "obj3", PredicateType.ON, negative=True)
    condition.add_edge("taxi0", "obj4", PredicateType.IN, negative=True)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a")
    state.add_node("b")
    state.add_node("c")
    state.add_node("d")
    state.add_node("e")
    state.add_node("f")

    state.add_edge("taxi0", "a", PredicateType.TOUCH_UP2D)
    state.add_edge("taxi0", "b", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d", PredicateType.TOUCH_DOWN2D)
    state.add_edge("taxi0", "e", PredicateType.ON, negative=False)
    state.add_edge("taxi0", "f", PredicateType.IN, negative=False)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_no_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")


if __name__ == "__main__":
    test_positive_positive()
    test_positive_negative()
