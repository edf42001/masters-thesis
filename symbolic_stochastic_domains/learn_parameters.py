import numpy as np
from typing import List, Tuple

from symbolic_stochastic_domains.learn_outcomes import covers

"""
Given a set of example data and ruleset with outcomes, learn the optimal probability distribution
to maximize the log likelihood of the result
"""


def example_covered_by(example, outcomes):
    """Returns all outcomes that cover an example"""
    # return [outcome for outcome in outcomes if covers(outcome, example)]
    return [outcome for outcome in outcomes if covers(outcome, example)]


def examples_covered_by_outcome(examples, outcome):
    return [ex for ex in examples if covers(outcome, ex)]


def likelihood_from_params_and_multiplier(params, multiplier) -> float:
    """
    Returns the log likelihood of the example set given the current params and the multiplier
    matrix which indicates which outcomes cover which examples
    """

    return np.sum(np.log(np.matmul(multiplier, params)))


def learn_params(examples, outcomes) -> Tuple[np.ndarray, float]:
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

    # Each example can be treated independently. The likelihood example is the total probability
    # of each outcome that covers it. This can be expressed with matrix multiplication
    # of a nxm matrix with ones where an outcome covers an example times an mx1 vector of outcome probs
    # Then the log of each row is taken individually to get the log likelihood
    # See section 5.1.2 of Learning Symbolic Models of Stochastic Domains

    # Initial param vector
    # params = np.ones((n_outcomes, 1), dtype=float) / n_outcomes
    params = np.array(params)
    params = params / np.sum(params)

    multiplier = np.zeros((n_examples, n_outcomes))
    for n, example in enumerate(examples):
        for m, outcome in enumerate(outcomes):
            multiplier[n, m] = covers(outcome, example)

    # print(params)
    # print(multiplier)
    params, likelihood = conditional_gradient_ascent(params, multiplier)

    # It tries to drive {} outcome to 0. Why is that? Because it is redundant? Is there a way for us to tell?
    # Because then we can remove that outcome

    return params, likelihood


def conditional_gradient_ascent(params: np.ndarray, multiplier: np.ndarray, epsilon=1E-4) -> Tuple[np.ndarray, float]:
    """
    Does conditional gradient ascent to find the maximum likelihood set of params
    Stops when change is less than epsilon
    """
    # Armijo rule step size parameters
    # TODO: I don't want to deal with this rule so what I do is if the grad goes down instead of up, decrease
    # This whole function is very inefficient
    # step size by b
    s = 1.0
    b = 0.5
    sigma = 0.01

    step_size = 0.05

    # Keep track of change in likelihood for stopping criteria
    last_l = -1E7
    current_l = -1E7+1

    while (current_l - last_l) > epsilon:
        start_likelihood = likelihood_from_params_and_multiplier(params, multiplier)

        gradients = []
        for i in range(len(params)):
            dx = 0.001
            params[i] += dx

            # There are faster ways than doing this whole multiplication again?
            new_likelihood = likelihood_from_params_and_multiplier(params, multiplier)

            params[i] -= dx

            grad = (new_likelihood - start_likelihood) / dx

            gradients.append(grad)

        # Bug where if there are two values that are symmetric, their values will be the same and the diff is 0
        # Get top two

        best_dir = np.argmax(gradients)
        best_grad = gradients[best_dir]

        # print(params.T, step_size, last_l - current_l, best_dir)

        # Store previous value of likelihood
        last_l = current_l

        # Update params and normalize
        backup_params = params.copy()

        params[best_dir] += step_size * best_grad
        params = params / np.sum(params)

        # TODO: This is silly and redundant
        current_l = likelihood_from_params_and_multiplier(params, multiplier)

        # This too, very inefficient
        while current_l < last_l:
            # Revert
            params = backup_params.copy()

            # Reduce step size
            step_size *= b

            # Try again
            params[best_dir] += step_size * best_grad
            params = params / np.sum(params)

            current_l = likelihood_from_params_and_multiplier(params, multiplier)

    return params, current_l

