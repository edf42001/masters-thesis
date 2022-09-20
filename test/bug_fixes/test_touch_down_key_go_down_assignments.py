"""
Created on 9/20/22 by Ethan Frank

For testing the following scenario:

>>>
2 applicable rules
Assignments for
Action 2:
{[~taxi0-TOUCH_DOWN2D-wall0, ~taxi0-TOUCH_DOWN2D-lock0]}
1.0: (<taxi.y, Increment(-1)>):
[~{'idpyo': 'lock'}]

Assignments for
Action 2:
{[taxi0-TOUCH_DOWN2D-lock0, lock0-OPEN-True]}
1.0: (<taxi.y, Increment(-1)>):
[{'idpyo': 'lock'}]

All possible assignments: [~{'idpyo': 'lock'} or {'idpyo': 'lock'}]
>>>

I think the correct outputs should be
[~{'idpyo': 'lock'}, ~{idpyo': 'wall'}] for the first and
[] for the second,

because the properties don't match, and either it being a wall or a lock
would have caused it to have not gone down
"""

from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from test.object_transfer.test_object_transfer_functions import determine_bindings_for_same_outcome

if __name__ == "__main__":
    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("wall0")
    condition.add_node("lock0")

    condition.add_edge("taxi0", "wall0", PredicateType.TOUCH_DOWN2D, negative=True)
    condition.add_edge("taxi0", "lock0", PredicateType.TOUCH_DOWN2D, negative=True)

    # The current state
    state = PredicateTree()
    state.add_node("taxi0")
    state.add_node("idpyo0")
    state.add_node("idpyo1")
    state.add_node("tyyaw0")

    state.add_edge("taxi0", "idpyo0", PredicateType.IN)
    state.add_edge("taxi0", "idpyo1", PredicateType.TOUCH_DOWN2D)
    state.add_edge("taxi0", "tyyaw0", PredicateType.TOUCH_LEFT2D)
    state.add_edge("taxi0", "tyyaw0", PredicateType.TOUCH_RIGHT2D)

    # For matching outcomes, determine which object must/must not be which
    assignments = determine_bindings_for_same_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")

    condition = PredicateTree()
    condition.add_node("taxi0")
    condition.add_node("wall0")
    condition.add_node("lock0")
    condition.add_edge("taxi0", "lock0", PredicateType.TOUCH_DOWN2D)
    condition.add_property("lock0", PredicateType.OPEN, True)

    assignments = determine_bindings_for_same_outcome(condition, state)

    print(f"Resulting Assignments: {assignments}")
