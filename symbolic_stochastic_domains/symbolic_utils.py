from typing import List

from symbolic_stochastic_domains.symbolic_classes import Outcome, Example, ExampleSet, Rule, RuleSet
from environment.symbolic_door_world import Predicate


def context_matches(context: List[Predicate], state: List[Predicate]) -> bool:
    """A context holds in a state if for every literal in the context, that literal has the same value in the state"""

    for literal in context:
        # This will match the type, object names, and equality using the Predicate's __eq__ functions
        if literal not in state:
            return False

    return True


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
    return outcome == example

def applicable(rule: Rule, example: Example) -> bool:
    """Returns if a rule is applicable to the example"""
    # A rule is applicable when a) the action matches, b) the context matches,
    # c) one of the outcomes of the rule covers the example's outcome

    return (
        rule.action == example.action and
        context_matches(rule.context, example.state) and
        any([covers(outcome, example) for outcome in rule.outcomes.outcomes])
    )


def examples_covered_by_outcome(outcome: Outcome, examples: ExampleSet) -> List[Example]:
    """Returns every example covered by the outcome"""
    return [ex for ex in examples.examples if covers(outcome, ex)]


def examples_applicable_by_rule(rule: Rule, examples: ExampleSet):
    """Returns every example that the rule is applicable to"""
    return [ex for ex in examples.examples if applicable(rule, ex)]


def proper_ruleset(rules: RuleSet):
    pass