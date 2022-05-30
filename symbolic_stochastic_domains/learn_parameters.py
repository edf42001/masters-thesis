from typing import List

from symbolic_stochastic_domains.symbolic_classes import ExampleSet, Rule, Outcome, Example
from symbolic_stochastic_domains.symbolic_utils import examples_applicable_by_rule, covers

"""
Given a set of example data and ruleset with outcomes, learn the optimal probability distribution
to maximize the log likelihood of the result
"""


def num_examples_covered_by_outcome(outcome: Outcome, applicable_examples: List[Example], examples: ExampleSet) -> int:
    """In the examples from the applicable example set, how many are covered by a specific outcome"""

    # We assume every example is covered by one and only one outcome
    # If the outcome covers the example, lookup in the example set dictionary how many times we observed that example
    return sum([examples.examples[example] for example in applicable_examples if covers(outcome, example)])


def learn_params(rule: Rule, examples: ExampleSet) -> List[float]:
    """
    LearnParameters takes an incomplete rule r consisting of an action,
    a set of deictic references, a context, and a set of outcomes,
    and learns the distribution P that maximizes r’s score on the examples E r covered by it
    """
    outcomes = rule.outcomes.outcomes
    num_outcomes = len(outcomes)

    probabilities = [0.0] * num_outcomes

    applicable_examples = examples_applicable_by_rule(rule, examples)
    total_applicable_examples = sum([examples.examples[example] for example in applicable_examples])

    print("Rule:")
    print(rule)
    print()
    print("Examples:")
    print(examples)
    print()
    print(f"Applicable Examples: ({total_applicable_examples})")
    for example in applicable_examples:
        print(example)
    print()

    # In this case where no outcomes overlap, the Maximum Likelihood is count based
    for i, outcome in enumerate(outcomes):
        num_covered = num_examples_covered_by_outcome(outcome, applicable_examples, examples)
        probabilities[i] = num_covered / total_applicable_examples
        print(f"{num_covered}: {outcome}")
    print()

    print(f"Final probabilities: {probabilities}")

    return probabilities
