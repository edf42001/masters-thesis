"""
Created on 10/22/22 by Ethan Frank

Tests for experience_helper.py.
"""

from symbolic_stochastic_domains.experience_helper import ExperienceHelper
from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.predicates_and_objects import PredicateType
from symbolic_stochastic_domains.symbolic_classes import Example

if __name__ == "__main__":
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
    state.add_edge("taxi0", "e0", PredicateType.ON)
    state.add_edge("taxi0", "f0", PredicateType.IN)
    state.add_property("d0", PredicateType.OPEN, True)

    experiences = ExperienceHelper.extract_experiences(state, n=2)

    for e in experiences:
        print(e)

    experience_helper = ExperienceHelper()

    example = Example(action=1, state=state, outcome=None)
    experience_helper.update_experience_dict(example)
    experience_helper.update_experience_dict(example)
    print(experience_helper.experiences_1)
