from typing import List
import math

from effects.effect import NoiseEffect
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
    return outcome == example.outcome


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


# TODO: This is silly because of all the cyclic imports. They should probably all be member variables
def score(rule: Rule, examples: ExampleSet) -> float:
    """
    Scores a rule on a set of examples. The score is the total likelihood - the penalty,
    where the penalty is the number of literals/effects in the outcomes and context of the rule
    This encourages simpler rules
    """

    # TODO: Should NoChange and Noise not be included in the penalty value?
    alpha = 0.5  # Penalty multiplier. Notice num atts in outcomes and len(self.context) are treated equally
    penalty = alpha * (rule.outcomes.get_total_num_affected_atts() + len(rule.context))

    # Approximate noise probability, used for calculating likelihood
    p_min = 0.01

    # For example, if outcome1 predicts 2 examples with probability 0.25, and outcome2 predicts
    # six examples with probability 0.75, the likelihood is 0.25^2 * 0.75^6.
    # Log likelihood of this is 0.25 * 2 + 0.75 * 6

    log_likelihood = 0

    # TODO:  Note, this is exactly the same computation done in learn_params. Should be someway to reuse that
    # I would like to use examples_applicable_by_rule and num_examples_covered_by_outcome, but this
    # creates cyclic imports. Perhaps this should be member functions instead of ones that take in args
    applicable_examples = examples_applicable_by_rule(rule, examples)

    # The probability this rule assigns to noise
    p_noise = 0
    for i, outcome in enumerate(rule.outcomes.outcomes):
        if type(outcome.outcome) is NoiseEffect:
            p_noise = rule.outcomes.probabilities[i]
            break

    for i, outcome in enumerate(rule.outcomes.outcomes):
        # TODO: noise outcomes num_covered need to be counted separately
        num_covered = sum([examples.examples[example] for example in applicable_examples if covers(outcome, example)])

        # Basic log of probability, but with the addition of the p_noise value * the p_min bound
        log_likelihood += math.log10(rule.outcomes.probabilities[i] + p_noise * p_min) * num_covered

    return log_likelihood - penalty


def ruleset_score(ruleset: RuleSet, examples: ExampleSet):
    """Total score of a ruleset is the sum of scores of rules in the ruleset"""
    return sum([score(rule, examples) for rule in ruleset.rules])

