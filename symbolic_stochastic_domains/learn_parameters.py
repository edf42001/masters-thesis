import numpy as np
from typing import List

from symbolic_stochastic_domains.learn_outcomes import covers

"""
Given a set of example data and ruleset with outcomes, learn the optimal probability distribution
to maximize the log likelihood of the result
"""


def example_covered_by(example, outcomes):
    """Returns all outcomes that cover an example"""
    return [outcome for outcome in outcomes if covers(outcome, example)]


def examples_covered_by_outcome(examples, outcome):
    return [ex for ex in examples if covers(outcome, ex)]


def learn_params(examples, outcomes) -> List[float]:
    n_outcomes = len(outcomes)
    n_examples = len(examples)

    params = []

    for example in examples:
        covered = example_covered_by(example, outcomes)

        print("Example {} covered by {}".format(example, covered))
    print()

    for outcome in outcomes:
        covered = examples_covered_by_outcome(examples, outcome)
        print("Outcome {} covers {}".format(outcome, covered))

        params.append(len(covered) / n_examples)
    print()

    # Do you add the probabilities of the ones that could have covered it to find the likelihood?
    # Example 4 covered by 0, 1, 2
    # Example 3 covered by 3

    likelihood()
    conditional_gradient_ascent()

    return params


def likelihood():
    params = [0.17, 0.17, 0.16, 0.5]

    l1 = (params[0] + params[1] + params[2])
    l2 = (params[3])
    l3 = (params[0])

    l = l1 * l2 * l3

    print("l1 {}".format(l1))
    print("l2 {}".format(l2))
    print("l3 {}".format(l3))
    print("l {}".format(l))


def conditional_gradient_ascent():
    params = [0.25, 0.25, 0.25, 0.25]

    for i in range(20):
        # derivative = 1.0 / (params[0] + params[1] + params[2]) + 1.0 / params[3]
        value = np.log(params[0] + params[1] + params[2]) + np.log(params[3]) + np.log(params[0]) + np.log(params[2])

        changes = []
        for i in range(len(params)):
            dx = 0.001
            params[i] += dx
            # new_derivative = 1.0 / (params[0] + params[1] + params[2]) + 1.0 / params[3]
            new_value = np.log(params[0] + params[1] + params[2]) + np.log(params[3]) + np.log(params[0]) + np.log(params[2])
            params[i] -= dx

            change = (new_value - value) / dx

            changes.append(change)

        # Bug where if there are two values that are symmetric, their values will be the same and the diff is 0
        # Get top two
        bests = np.argpartition(changes, -2)[-2:]
        best_dir = bests[-1]
        max_grad = changes[best_dir]
        second_max_grad = changes[bests[-2]]

        diff = max_grad - second_max_grad
        print(changes, best_dir, max_grad, diff)
        params[best_dir] += 0.15 * diff
        params = [p / sum(params) for p in params]
        print(params)
        print(value)

    # Maximize (0 + 1 + 2) * (0) * (3) -> ln(0+1+2) + ln(0) + ln(3)
    # Subject to (0 + 1 + 2 + 3) = 1
    # and all greater than 0

    # Maximize (0 + 1) * 0
    # Subject to 0 + 1 = 1



