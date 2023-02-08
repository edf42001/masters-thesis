from typing import List, Set, Dict

from symbolic_stochastic_domains.symbolic_classes import Outcome, Example, Rule
from symbolic_stochastic_domains.predicate_tree import PredicateTree


def context_matches(context: PredicateTree, state: PredicateTree) -> bool:
    """
    A context holds in a state if there is a match for all of the edges/nodes in the context
    and no negative edges match (A negative edge is like a false literal)
    """

    # Need to check that there exists the context tree contained in the state tree
    # This could be framed recursively by saying there is a node in the context, and all of it's edges
    # are in the state, and all of the edges nodes are contained in the state's edges' nodes
    return state.base_object.contains(context.base_object)


def covers(outcome: Outcome, example: Example) -> bool:
    """
    Returns true if the outcome covers the example
    Meaning, when you apply the outcome to the initial state of the example,
    does the end state equal the final state?
    """

    # TODO: Could take into account different effects that lead to same result, such as SetToTrue and NoChange
    # Having the same outcome when the value is already true. Otherwise will need to treat each case separately

    # Check that every outcome in the outcome, and that none are in the example that aren't in the outcome

    # I think for now, this is the same as saying they are exactly equal
    # Eventually it may need to be if initial state variable with outcome applied == final state variable.
    return outcome == example.outcome


def applicable(rule: Rule, example: Example) -> bool:
    """Returns if a rule is applicable to the example"""
    # A rule is applicable when a) the action matches, b) the context matches,
    # c) one of the outcomes of the rule covers the example's outcome

    return (
        rule.action == example.action and
        # By switching the order of the last two here, we can cause one or other other to run more or less
        context_matches(rule.context, example.state) and
        any([covers(outcome, example) for outcome in rule.outcomes.outcomes])
    )
