from typing import Tuple

from symbolic_stochastic_domains.state import State


def covers(outcome: dict, example: Tuple[State, State]) -> bool:
    """
    Returns true if the outcome covers the example
    Meaning, when you apply the outcome to the initial state of the example,
    does the end state equal the final state?
    """
    s1 = example[0]
    s2 = example[1]

    # Go through every state variable. If they differ, see if the outcome would fix that
    # Otherwise, this outcome does not cover this
    for term1, term2 in zip(s1, s2):
        if term1.get_negated() != term2.get_negated():  # If conflict
            name = term2.get_unique_id()
            if name not in outcome or outcome[name] != term2.true():  # Does an outcome not cover this?
                return False

    # Also, need to check the outcome doesn't undo anything
    # Some redundant checks in here
    for term, truth in outcome.items():
        if term in s2 and s2[term] != truth:
            return False

    return True