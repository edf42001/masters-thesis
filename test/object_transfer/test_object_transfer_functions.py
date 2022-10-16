"""
Created on 7/30/22 by Ethan Frank

Test low level functions for object knowledge transfer with specific cases
"""


from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from symbolic_stochastic_domains.object_transfer import determine_bindings_for_same_outcome, determine_bindings_for_no_outcome


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

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_UP)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c0", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d0", PredicateType.TOUCH_DOWN)
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

    state.add_edge("taxi0", "a0", PredicateType.TOUCH_UP)
    state.add_edge("taxi0", "b0", PredicateType.TOUCH_LEFT)
    state.add_edge("taxi0", "c0", PredicateType.TOUCH_RIGHT)
    state.add_edge("taxi0", "d0", PredicateType.TOUCH_DOWN)
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
