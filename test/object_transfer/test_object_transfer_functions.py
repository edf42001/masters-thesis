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
        assert unknown not in self.positives, "One object can never be two different objects"
        self.positives[unknown] = known

    def add_negative(self, unknown, known):
        # Use lists here because it is possible that 1 unknown is not two different objects
        if unknown not in self.negatives:
            self.negatives[unknown] = [known]
        else:
            self.negatives[unknown].append(known)

    def dict_to_str(self, dictionary):
        return str({key: ", ".join(values) for key, values in dictionary.items()})

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if len(self.positives) > 0 and len(self.negatives) > 0:
            return str(self.positives) + " - ~" + self.dict_to_str(self.negatives)
        elif len(self.negatives) > 0:
            return "~" + self.dict_to_str(self.negatives)
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

    def add_assignments(self, assignment_list):
        self.assignments.extend(assignment_list.assignments)
        self.hash = hash(self.__str__())

    def __str__(self):
        return "[" + " or ".join(str(a) for a in self.assignments) + "]"

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

    # Make sure to remove the number from all the object names

    assignment = ObjectAssignment()

    # First, check all positive edges. All of these must match
    # However, note that if any properties of the positive edges don't match, this was not the real rule
    # That caused this to occur. So, we return None just like below, and hope that some other rule covered
    # this occurrence.
    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same

                # Check if any of the properties do not match.
                for prop, value in edge.to_node.properties.items():
                    # If the property is not present, assume it is false.
                    # Then, I don't have to add false properties to every object, but the learner can't cheat
                    # by using which properties are present to distinguish objects
                    other_value = False if prop not in edge2.to_node.properties else edge2.to_node.properties[prop]

                    if value != other_value:
                        return None

                assignment.add_positive(edge2.to_node.object_name[:-1], edge.to_node.object_name[:-1])
                found = True
                break

        # If this happens, it is because this action has more than one rule that covers it,
        # and this condition was not the one. This is fine, the other condition will cover it.
        # If neither cover it, then we have a paradox in the code.
        if not found:
            return None

    # Now all negative edges, any matches here must be false
    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment.add_negative(edge2.to_node.object_name[:-1], edge.to_node.object_name[:-1])
                break

    # For later down the line, it'll be easier if the list is completely empty, instead of having and empty object
    if len(assignment.negatives) == 0 and len(assignment.positives) == 0:
        return ObjectAssignmentList([])
    else:
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

    # The same is true of properties. If any of the properties of positive edges do not match, that could be
    # why nothing happened. So we can't really give any more information.

    for edge in condition.base_object.edges:
        # Find matching edge
        found = False
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                # This is the match, these objects are the same

                # Check if any of the properties do not match.
                for prop, value in edge.to_node.properties.items():
                    # If the property is not present, assume it is false.
                    # Then, I don't have to add false properties to every object, but the learner can't cheat
                    # by using which properties are present to distinguish objects
                    other_value = False if prop not in edge2.to_node.properties else edge2.to_node.properties[prop]

                    if value != other_value:
                        return ObjectAssignmentList([])

                found = True
                break

        if not found:
            return ObjectAssignmentList([])

    # Now, let's look for negative edges that perhaps are causing the issue
    # Also some positive edges could be missing. Each of these are independent, add them to the array separately.
    for edge in condition.base_object.edges:
        for edge2 in state.base_object.edges:  # (Could probably combine this with the above, then throw out if bad?)
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_negative(edge2.to_node.object_name[:-1], edge.to_node.object_name[:-1])
                assignments.append(assignment)
                break

    for edge in condition.base_object.negative_edges:
        for edge2 in state.base_object.edges:
            if edge.type == edge2.type:
                assignment = ObjectAssignment()
                assignment.add_positive(edge2.to_node.object_name[:-1], edge.to_node.object_name[:-1])
                assignments.append(assignment)
                break

    return ObjectAssignmentList(assignments)


def test_positive_positive():
    # The rule's condition
    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obja1")  # We append numbers on the end because it strips those off to get the object name
    condition.add_node("objb2")
    condition.add_node("objc3")
    condition.add_node("objd4")

    condition.add_edge("taxi0", "obja1", PredicateType.TOUCH_LEFT)
    condition.add_edge("taxi0", "objb2", PredicateType.TOUCH_RIGHT)
    condition.add_edge("taxi0", "objc3", PredicateType.ON, negative=True)
    condition.add_edge("taxi0", "objd4", PredicateType.IN, negative=True)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a0")
    state.add_node("b0")
    state.add_node("c0")
    state.add_node("d0")
    state.add_node("e0")
    state.add_node("f0")

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_UP2D)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c0", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d0", PredicateType.TOUCH_DOWN2D)
    state.add_edge("taxi0", "e0", PredicateType.ON, negative=False)
    state.add_edge("taxi0", "f0", PredicateType.IN, negative=False)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_same_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")


def test_positive_negative():
    # The rule's condition
    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obja1")
    condition.add_node("objb2")
    condition.add_node("objc3")
    condition.add_node("objd4")

    condition.add_edge("taxi0", "obja1", PredicateType.TOUCH_LEFT)
    condition.add_edge("taxi0", "objb2", PredicateType.TOUCH_RIGHT)
    condition.add_edge("taxi0", "objc3", PredicateType.ON, negative=True)
    condition.add_edge("taxi0", "objd4", PredicateType.IN, negative=True)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a0")
    state.add_node("b0")
    state.add_node("c0")
    state.add_node("d0")
    state.add_node("e0")
    state.add_node("f0")

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_UP2D)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c0", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d0", PredicateType.TOUCH_DOWN2D)
    state.add_edge("taxi0", "e0", PredicateType.ON, negative=False)
    state.add_edge("taxi0", "f0", PredicateType.IN, negative=False)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_no_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")


def test_positive_positive_with_properties():
    # If we observed the outcome that we predicted, we don't have to care about the properties
    # This is because only one object can be referenced per fluent per timestep,
    # so if there was an object where the rule said it wasn, then the properties must be correct,
    # No matter what

    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obja1")
    condition.add_node("objb2")

    condition.add_edge("taxi0", "obja1", PredicateType.TOUCH_LEFT)
    condition.add_property("obja1", PredicateType.OPEN, True)
    condition.add_edge("taxi0", "objb2", PredicateType.TOUCH_RIGHT)
    condition.add_property("objb2", PredicateType.OPEN, False)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a0")
    state.add_node("b0")

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_LEFT)
    state.add_property("a0", PredicateType.OPEN, True)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_RIGHT)
    state.add_property("b0", PredicateType.OPEN, False)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_same_outcome(condition, state)
    print(f"Resulting Assignments: {assignments}")


def test_positive_negative_with_properties():
    # Now in this case, we have to modify our algorithm a bit.
    # If we expected to have a object with a certain property, and the property differs,
    # then maybe the object is actually the correct object but the issue was the property was wrong
    # Should we return both "nothing" as a possible assignment, and that the object was wrong?
    # Or because we can't say which it is, just one? I think we have to return all possibilities that
    # could have caused it to fail. Because I say "at least one of these needs to be correct"
    # Or do we just return empty?

    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("obja1")
    condition.add_node("objb2")

    condition.add_edge("taxi0", "obja1", PredicateType.TOUCH_LEFT)
    condition.add_property("obja1", PredicateType.OPEN, True)
    condition.add_edge("taxi0", "objb2", PredicateType.TOUCH_RIGHT)
    condition.add_property("objb2", PredicateType.OPEN, False)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("a0")
    state.add_node("b0")

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_LEFT)
    state.add_property("a0", PredicateType.OPEN, True)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_RIGHT)
    state.add_property("b0", PredicateType.OPEN, True)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_no_outcome(condition, state)
    print(f"Resulting Assignments: {assignments}")


if __name__ == "__main__":
    test_positive_positive()
    test_positive_negative()
    test_positive_positive_with_properties()
    test_positive_negative_with_properties()
